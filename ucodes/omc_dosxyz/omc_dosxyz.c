/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations
 
 Copyright (C) 2018 Edgardo Doerner (edoerner@fis.puc.cl)


 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
*****************************************************************************/

/******************************************************************************
 omc_dosxyz - An ompMC user code to calculate deposited dose on voxelized 
 geometries.  
*****************************************************************************/

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "omc_utilities.h"
#include "ompmc.h"
#include "omc_random.h"

/******************************************************************************/
/* Parsing program options with getopt long
 http://www.gnu.org/software/libc/manual/html_node/Getopt.html#Getopt */
#include <getopt.h>

/******************************************************************************/
/* Geometry definitions */
struct Geom {
    int *med_indices;           // index of the media in each voxel
    double *med_densities;      // density of the medium in each voxel
    
    int isize;                  // number of voxels on each direction
    int jsize;
    int ksize;
    
    double *xbounds;            // boundaries of voxels on each direction
    double *ybounds;
    double *zbounds;
};
struct Geom geometry;

void initPhantom() {
    
    /* Get phantom file path from input data */
    char phantom_file[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "phantom file") != 1) {
        printf("Can not find 'phantom file' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(phantom_file, buffer);
    
    /* Open .egsphant file */
    FILE *fp;
    
    if ((fp = fopen(phantom_file, "r")) == NULL) {
        printf("Unable to open file: %s\n", phantom_file);
        exit(EXIT_FAILURE);
    }
    
    printf("Path to phantom file : %s\n", phantom_file);
    
    /* Get number of media in the phantom */
    fgets(buffer, BUFFER_SIZE, fp);
    media.nmed = atoi(buffer);
    
    /* Get media names on phantom file */
    for (int i=0; i<media.nmed; i++) {
        fgets(buffer, BUFFER_SIZE, fp);
        removeSpaces(media.med_names[i], buffer);
    }
    
    /* Skip next line, it contains dummy input */
    fgets(buffer, BUFFER_SIZE, fp);
    
    /* Read voxel numbers on each direction */
    fgets(buffer, BUFFER_SIZE, fp);
    sscanf(buffer, "%d %d %d", &geometry.isize,
           &geometry.jsize, &geometry.ksize);
    
    /* Read voxel boundaries on each direction */
    geometry.xbounds = malloc((geometry.isize + 1)*sizeof(double));
    geometry.ybounds = malloc((geometry.jsize + 1)*sizeof(double));
    geometry.zbounds = malloc((geometry.ksize + 1)*sizeof(double));
    
    for (int i=0; i<=geometry.isize; i++) {
        fscanf(fp, "%lf", &geometry.xbounds[i]);
    }
    for (int i=0; i<=geometry.jsize; i++) {
        fscanf(fp, "%lf", &geometry.ybounds[i]);
     }
    for (int i=0; i<=geometry.ksize; i++) {
        fscanf(fp, "%lf", &geometry.zbounds[i]);
    }
    
    /* Skip the rest of the last line read before */
    fgets(buffer, BUFFER_SIZE, fp);
    
    /* Read media indices */
    int irl = 0;    // region index
    char idx;
    geometry.med_indices =
        malloc(geometry.isize*geometry.jsize*geometry.ksize*sizeof(int));
    for (int k=0; k<geometry.ksize; k++) {
        for (int j=0; j<geometry.jsize; j++) {
            for (int i=0; i<geometry.isize; i++) {
                irl = i + j*geometry.isize + k*geometry.jsize*geometry.isize;
                idx = fgetc(fp);
                /* Convert digit stored as char to int */
                geometry.med_indices[irl] = idx - '0';
            }
            /* Jump to next line */
            fgets(buffer, BUFFER_SIZE, fp);
        }
        /* Skip blank line */
        fgets(buffer, BUFFER_SIZE, fp);
    }
    
    /* Read media densities */
    geometry.med_densities =
        malloc(geometry.isize*geometry.jsize*geometry.ksize*sizeof(double));
    for (int k=0; k<geometry.ksize; k++) {
        for (int j=0; j<geometry.jsize; j++) {
            for (int i=0; i<geometry.isize; i++) {
                irl = i + j*geometry.isize + k*geometry.jsize*geometry.isize;
                fscanf(fp, "%lf", &geometry.med_densities[irl]);
            }
        }
        /* Skip blank line */
        fgets(buffer, BUFFER_SIZE, fp);
    }
    
    /* Summary with geometry information */
    printf("Number of media in phantom : %d\n", media.nmed);
    printf("Media names: ");
    for (int i=0; i<media.nmed; i++) {
        printf("%s, ", media.med_names[i]);
    }
    printf("\n");
    printf("Number of voxels on each direction (X,Y,Z) : (%d, %d, %d)\n",
           geometry.isize, geometry.jsize, geometry.ksize);
    printf("Minimum and maximum boundaries on each direction : \n");
    printf("\tX (cm) : %lf, %lf\n",
           geometry.xbounds[0], geometry.xbounds[geometry.isize]);
    printf("\tY (cm) : %lf, %lf\n",
           geometry.ybounds[0], geometry.ybounds[geometry.jsize]);
    printf("\tZ (cm) : %lf, %lf\n",
           geometry.zbounds[0], geometry.zbounds[geometry.ksize]);
    
    /* Close phantom file */
    fclose(fp);
    
    return;
}

void cleanPhantom() {
    
    free(geometry.xbounds);
    free(geometry.ybounds);
    free(geometry.zbounds);
    free(geometry.med_indices);
    free(geometry.med_densities);
    return;
}

void howfar(int *idisc, int *irnew, double *ustep) {
    
    int np = stack.np;
    int irl = stack.ir[np];
    double dist = 0.0;
    
    if (stack.ir[np] == 0) {
        /* The particle is outside the geometry, terminate history */
        *idisc = 1;
        return;
    }
    
    /* If here, the particle is in the geometry, do transport checks */
    int ijmax = geometry.isize*geometry.jsize;
    int imax = geometry.isize;
    
    /* First we need to decode the region number of the particle in terms of
     the region indices in each direction */
    int irx = (irl - 1)%imax;
    int irz = (irl - 1 - irx)/ijmax;
    int iry = ((irl - 1 - irx) - irz*ijmax)/imax;
    
    /* Check in z-direction */
    if (stack.w[np] > 0.0) {
        /* Going towards outer plane */
        dist = (geometry.zbounds[irz+1] - stack.z[np])/stack.w[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irz != (geometry.ksize - 1)) {
                *irnew = irl + ijmax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    else if (stack.w[np] < 0.0) {
        /* Going towards inner plane */
        dist = -(stack.z[np] - geometry.zbounds[irz])/stack.w[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irz != 0) {
                *irnew = irl - ijmax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }

    /* Check in x-direction */
    if (stack.u[np] > 0.0) {
        /* Going towards positive plane */
        dist = (geometry.xbounds[irx+1] - stack.x[np])/stack.u[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irx != (geometry.isize - 1)) {
                *irnew = irl + 1;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    else if (stack.u[np] < 0.0) {
        /* Going towards negative plane */
        dist = -(stack.x[np] - geometry.xbounds[irx])/stack.u[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irx != 0) {
                *irnew = irl - 1;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    /* Check in y-direction */
    if (stack.v[np] > 0.0) {
        /* Going towards positive plane */
        dist = (geometry.ybounds[iry+1] - stack.y[np])/stack.v[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (iry != (geometry.jsize - 1)) {
                *irnew = irl + imax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    else if (stack.v[np] < 0.0) {
        /* Going towards negative plane */
        dist = -(stack.y[np] - geometry.ybounds[iry])/stack.v[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (iry != 0) {
                *irnew = irl - imax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    return;
}

double hownear(void) {
    
    int np = stack.np;
    int irl = stack.ir[np];
    double tperp = 1.0E10;  /* perpendicular distance to closest boundary */
    
    if (irl == 0) {
        /* Particle exiting geometry */
        tperp = 0.0;
    }
    else {
        /* In the geometry, do transport checks */
        int ijmax = geometry.isize*geometry.jsize;
        int imax = geometry.isize;
        
        /* First we need to decode the region number of the particle in terms
         of the region indices in each direction */
        int irx = (irl - 1)%imax;
        int irz = (irl - 1 - irx)/ijmax;
        int iry = ((irl - 1 - irx) - irz*ijmax)/imax;
        
        /* Check in x-direction */
        tperp = fmin(tperp, geometry.xbounds[irx+1] - stack.x[np]);
        tperp = fmin(tperp, stack.x[np] - geometry.xbounds[irx]);
        
        /* Check in y-direction */
        tperp = fmin(tperp, geometry.ybounds[iry+1] - stack.y[np]);
        tperp = fmin(tperp, stack.y[np] - geometry.ybounds[iry]);
        
        /* Check in z-direction */
        tperp = fmin(tperp, geometry.zbounds[irz+1] - stack.z[np]);
        tperp = fmin(tperp, stack.z[np] - geometry.zbounds[irz]);
    }
    
    return tperp;
}
/******************************************************************************/

/******************************************************************************/
/* Source definitions */
const int MXEBIN = 200;     // number of energy bins of spectrum
const int INVDIM = 1000;    // number of bins in inverse CDF

struct Source {
    int spectrum;               // 0 : monoenergetic, 1 : spectrum
    int charge;                 // 0 : photons, -1 : electron, +1 : positron
    
    /* For monoenergetic source */
    double energy;
    
    /* For spectrum */
    double deltak;              // number of elements in inverse CDF
    double *cdfinv1;            // energy value of bin
    double *cdfinv2;            // prob. that particle has energy xi
    
    /* Source shape information */
    double ssd;                 // distance of point source to phantom surface
    double xinl, xinu;          // lower and upper x-bounds of the field on
                                // phantom surface
    double yinl, yinu;          // lower and upper y-bounds of the field on
                                // phantom surface
    double xsize, ysize;        // x- and y-width of collimated field
    int ixinl, ixinu;        // lower and upper x-bounds indices of the
                                // field on phantom surface
    int iyinl, iyinu;        // lower and upper y-bounds indices of the
                                // field on phantom surface
};
struct Source source;

void initSource() {
    
    /* Get spectrum file path from input data */
    char spectrum_file[128];
    char buffer[BUFFER_SIZE];
    
    source.spectrum = 1;    /* energy spectrum as default case */
    
    /* First check of spectrum file was given as an input */
    if (getInputValue(buffer, "spectrum file") != 1) {
        printf("Can not find 'spectrum file' key on input file.\n");
        printf("Switch to monoenergetic case.\n");
        source.spectrum = 0;    /* monoenergetic source */
    }
    
    if (source.spectrum) {
        removeSpaces(spectrum_file, buffer);
        
        /* Open .source file */
        FILE *fp;
        
        if ((fp = fopen(spectrum_file, "r")) == NULL) {
            printf("Unable to open file: %s\n", spectrum_file);
            exit(EXIT_FAILURE);
        }
        
        printf("Path to spectrum file : %s\n", spectrum_file);
        
        /* Read spectrum file title */
        fgets(buffer, BUFFER_SIZE, fp);
        printf("Spectrum file title: %s", buffer);
        
        /* Read number of bins and spectrum type */
        double enmin;   /* lower energy of first bin */
        int nensrc;     /* number of energy bins in spectrum histogram */
        int imode;      /* 0 : histogram counts/bin, 1 : counts/MeV*/
        
        fgets(buffer, BUFFER_SIZE, fp);
        sscanf(buffer, "%d %lf %d", &nensrc, &enmin, &imode);
        
        if (nensrc > MXEBIN) {
            printf("Number of energy bins = %d is greater than max allowed = "
                   "%d. Increase MXEBIN macro!\n", nensrc, MXEBIN);
            exit(EXIT_FAILURE);
        }
        
        /* upper energy of bin i in MeV */
        double *ensrcd = malloc(nensrc*sizeof(double));
        /* prob. of finding a particle in bin i */
        double *srcpdf = malloc(nensrc*sizeof(double));
        
        /* Read spectrum information */
        for (int i=0; i<nensrc; i++) {
            fgets(buffer, BUFFER_SIZE, fp);
            sscanf(buffer, "%lf %lf", &ensrcd[i], &srcpdf[i]);
        }
        printf("Have read %d input energy bins from spectrum file.\n", nensrc);
        
        if (imode == 0) {
            printf("Counts/bin assumed.\n");
        }
        else if (imode == 1) {
            printf("Counts/MeV assumed.\n");
            srcpdf[0] *= (ensrcd[0] - enmin);
            for(int i=1; i<nensrc; i++) {
                srcpdf[i] *= (ensrcd[i] - ensrcd[i - 1]);
            }
        }
        else {
            printf("Invalid mode number in spectrum file.");
            exit(EXIT_FAILURE);
        }
        
        double ein = ensrcd[nensrc - 1];
        printf("Energy ranges from %f to %f MeV\n", enmin, ein);
        
        /* Initialization routine to calculate the inverse of the
         cumulative probability distribution that is used during execution to
         sample the incident particle energy. */
        double *srccdf = malloc(nensrc*sizeof(double));
        
        srccdf[0] = srcpdf[0];
        for (int i=1; i<nensrc; i++) {
            srccdf[i] = srccdf[i-1] + srcpdf[i];
        }
        
        double fnorm = 1.0/srccdf[nensrc - 1];
        double binsok = 0.0;
        source.deltak = INVDIM; /* number of elements in inverse CDF */
        double gridsz = 1.0f/source.deltak;
        
        for (int i=0; i<nensrc; i++) {
            srccdf[i] *= fnorm;
            if (i == 0) {
                if (srccdf[0] <= 3.0*gridsz) {
                    binsok = 1.0;
                }
            }
            else {
                if ((srccdf[i] - srccdf[i - 1]) < 3.0*gridsz) {
                    binsok = 1.0;
                }
            }
        }
        
        if (binsok != 0.0) {
            printf("Warning!, some of normalized bin probabilities are "
                   "so small that bins may be missed.\n");
        }

        /* Calculate cdfinv. This array allows the rapid sampling for the
         energy by precomputing the results for a fine grid. */
        source.cdfinv1 = malloc(source.deltak*sizeof(double));
        source.cdfinv2 = malloc(source.deltak*sizeof(double));
        double ak;
        
        for (int k=0; k<source.deltak; k++) {
            ak = (double)k*gridsz;
            int i;
            
            for (i=0; i<nensrc; i++) {
                if (ak <= srccdf[i]) {
                    break;
                }
            }
            
            /* We should fall here only through the above break sentence. */
            if (i != 0) {
                source.cdfinv1[k] = ensrcd[i - 1];
            }
            else {
                source.cdfinv1[k] = enmin;
            }
            source.cdfinv2[k] = ensrcd[i] - source.cdfinv1[k];
            
        }
        
        /* Cleaning */
        fclose(fp);
        free(ensrcd);
        free(srcpdf);
        free(srccdf);
    }
    else {  /* monoenergetic source */
        if (getInputValue(buffer, "mono energy") != 1) {
            printf("Can not find 'mono energy' key on input file.\n");
            exit(EXIT_FAILURE);
        }
        source.energy = atof(buffer);
        printf("%f monoenergetic source\n", source.energy);
        
    }
    
    /* Initialize geometrical data of the source */
    
    /* Read collimator rectangle */
    if (getInputValue(buffer, "collimator bounds") != 1) {
        printf("Can not find 'collimator bounds' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    sscanf(buffer, "%lf %lf %lf %lf", &source.xinl,
           &source.xinu, &source.yinl, &source.yinu);
    
    /* Calculate x-direction input zones */
    if (source.xinl < geometry.xbounds[0]) {
        source.xinl = geometry.xbounds[0];
    }
    if (source.xinu <= source.xinl) {
        source.xinu = source.xinl;  /* default a pencil beam */
    }
    
    /* Check radiation field is not too big against the phantom */
    if (source.xinu > geometry.xbounds[geometry.isize]) {
        source.xinu = geometry.xbounds[geometry.isize];
    }
    if (source.xinl > geometry.xbounds[geometry.isize]) {
        source.xinl = geometry.xbounds[geometry.isize];
    }
    
    /* Now search for initial region x index range */
    printf("Index ranges for radiation field:\n");
    source.ixinl = 0;
    while ((geometry.xbounds[source.ixinl] <= source.xinl) &&
           (geometry.xbounds[source.ixinl + 1] < source.xinl)) {
        source.ixinl++;
    }
        
    source.ixinu = source.ixinl - 1;
    while ((geometry.xbounds[source.ixinu] <= source.xinu) &&
           (geometry.xbounds[source.ixinu + 1] < source.xinu)) {
        source.ixinu++;
    }
    printf("i index ranges over i = %d to %d\n", source.ixinl, source.ixinu);
    
    /* Calculate y-direction input zones */
    if (source.yinl < geometry.ybounds[0]) {
        source.yinl = geometry.ybounds[0];
    }
    if (source.yinu <= source.yinl) {
        source.yinu = source.yinl;  /* default a pencil beam */
    }
    
    /* Check radiation field is not too big against the phantom */
    if (source.yinu > geometry.ybounds[geometry.jsize]) {
        source.yinu = geometry.ybounds[geometry.jsize];
    }
    if (source.yinl > geometry.ybounds[geometry.jsize]) {
        source.yinl = geometry.ybounds[geometry.jsize];
    }
    
    /* Now search for initial region y index range */
    source.iyinl = 0;
    while ((geometry.ybounds[source.iyinl] <= source.yinl) &&
           (geometry.ybounds[source.iyinl + 1] < source.yinl)) {
        source.iyinl++;
    }
    source.iyinu = source.iyinl - 1;
    while ((geometry.ybounds[source.iyinu] <= source.yinu) &&
           (geometry.ybounds[source.iyinu + 1] < source.yinu)) {
        source.iyinu++;
    }
    printf("j index ranges over i = %d to %d\n", source.iyinl, source.iyinu);

    /* Calculate collimator sizes */
    source.xsize = source.xinu - source.xinl;
    source.ysize = source.yinu - source.yinl;
    
    /* Read source charge */
    if (getInputValue(buffer, "charge") != 1) {
        printf("Can not find 'charge' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    
    source.charge = atoi(buffer);
    if (source.charge < -1 || source.charge > 1) {
        printf("Particle kind not recognized.\n");
        exit(EXIT_FAILURE);
    }
    
    /* Read source SSD */
    if (getInputValue(buffer, "ssd") != 1) {
        printf("Can not find 'ssd' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    
    source.ssd = atof(buffer);
    if (source.ssd < 0) {
        printf("SSD must be greater than zero.\n");
        exit(EXIT_FAILURE);
    }
    
    /* Print some information for debugging purposes */
    if (verbose_flag) {
        printf("Source information :\n");
        printf("\t Charge = %d\n", source.charge);
        printf("\t SSD (cm) = %f\n", source.ssd);
        printf("Collimator :\n");
        printf("\t x (cm) : min = %f, max = %f\n", source.xinl, source.xinu);
        printf("\t y (cm) : min = %f, max = %f\n", source.yinl, source.yinu);
        printf("Sizes :\n");
        printf("\t x (cm) = %f, y (cm) = %f\n", source.xsize, source.ysize);
    }
    
    return;
}

void cleanSource() {
    
    free(source.cdfinv1);
    free(source.cdfinv2);
    
    return;
}

/******************************************************************************/
/* Scoring definitions */
struct Score {
    double ensrc;               // total energy from source
    double *endep;              // 3D dep. energy matrix per batch
    
    /* The following variables are needed for statistical analysis. Their
     values are accumulated across the simulation */
    double *accum_endep;        // 3D deposited energy matrix
    double *accum_endep2;       // 3D square deposited energy
};
struct Score score;

void initScore() {
    
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    score.ensrc = 0.0;
    
    /* Region with index 0 corresponds to region outside phantom */
    score.endep = malloc((gridsize + 1)*sizeof(double));
    score.accum_endep = malloc((gridsize + 1)*sizeof(double));
    score.accum_endep2 = malloc((gridsize + 1)*sizeof(double));
    
    /* Initialize all arrays to zero */
    memset(score.endep, 0.0, (gridsize + 1)*sizeof(double));
    memset(score.accum_endep, 0.0, (gridsize + 1)*sizeof(double));
    memset(score.accum_endep2, 0.0, (gridsize + 1)*sizeof(double));
    
    return;
}

void cleanScore() {
    
    free(score.endep);
    free(score.accum_endep);
    free(score.accum_endep2);
    
    return;
}

void ausgab(double edep) {
    
    int np = stack.np;
    int irl = stack.ir[np];
    double endep = stack.wt[np]*edep;
        
    /* Deposit particle energy on spot */
    #pragma omp atomic
    score.endep[irl] += endep;
    
    return;
}

void accumEndep() {
    
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    /* Accumulate endep and endep squared for statistical analysis */
    double edep = 0.0;
    
    int irl = 0;
    
    #pragma omp parallel for firstprivate(edep)
    for (irl=0; irl<gridsize + 1; irl++) {
        edep = score.endep[irl];
        
        score.accum_endep[irl] += edep;
        score.accum_endep2[irl] += edep*edep;
    }
    
    /* Clean scoring array */
    memset(score.endep, 0.0, (gridsize + 1)*sizeof(double));
    
    return;
}

void accumulateResults(int iout, int nhist, int nbatch)
{
    int irl;
    int imax = geometry.isize;
    int ijmax = geometry.isize*geometry.jsize;
    double endep, endep2, unc_endep;

    /* Calculate incident fluence */
    double inc_fluence = (double)nhist;
    double mass;
    int iz;

    #pragma omp parallel for private(irl,endep,endep2,unc_endep,mass)
    for (iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                endep = score.accum_endep[irl];
                endep2 = score.accum_endep2[irl];
                
                /* First calculate mean deposited energy across batches and its
                 uncertainty */
                endep /= (double)nbatch;
                endep2 /= (double)nbatch;
                
                /* Batch approach uncertainty calculation */
                if (endep != 0.0) {
                    unc_endep = endep2 - endep*endep;
                    unc_endep /= (double)(nbatch - 1);
                    
                    /* Relative uncertainty */
                    unc_endep = sqrt(unc_endep)/endep;
                }
                else {
                    endep = 0.0;
                    unc_endep = 0.9999999;
                }
                
                /* We separate de calculation of dose, to give the user the
                 option to output mean energy (iout=0) or deposited dose
                 (iout=1) per incident fluence */
                
                if (iout) {
                    
                    /* Convert deposited energy to dose */
                    mass = (geometry.xbounds[ix+1] - geometry.xbounds[ix])*
                        (geometry.ybounds[iy+1] - geometry.ybounds[iy])*
                        (geometry.zbounds[iz+1] - geometry.zbounds[iz]);
                    
                    /* Transform deposited energy to Gy */
                    mass *= geometry.med_densities[irl-1];
                    endep *= 1.602E-10/(mass*inc_fluence);
                    
                } else {    /* Output mean deposited energy */
                    endep /= inc_fluence;
                }
                
                /* Store output quantities */
                score.accum_endep[irl] = endep;
                score.accum_endep2[irl] = unc_endep;
            }
        }
    }
    
    /* Zero dose in air */
    #pragma omp parallel for private(irl)
    for (iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                
                if(geometry.med_densities[irl-1] < 0.044) {
                    score.accum_endep[irl] = 0.0;
                    score.accum_endep2[irl] = 0.9999999;
                }
            }
        }
    }
    
    return;
}

void outputResults(char *output_file, int iout, int nhist, int nbatch) {
    
    // Accumulate the results
    accumulateResults(iout, nhist,nbatch);
    
    int irl;
    int imax = geometry.isize;
    int ijmax = geometry.isize*geometry.jsize;
    
    /* Output to file */
    char extension[15];
    if (iout) {
        strcpy(extension, ".3ddose");
    } else {
        strcpy(extension, ".3denergy");
    }
    
    /* Get file path from input data */
    char output_folder[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "output folder") != 1) {
        printf("Can not find 'output folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(output_folder, buffer);
    
    /* Make space for the new string */
    char* file_name = malloc(strlen(output_folder) + strlen(output_file) + 
        strlen(extension) + 1);
    strcpy(file_name, output_folder);
    strcat(file_name, output_file); /* add the file name */
    strcat(file_name, extension); /* add the extension */
    
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
    /* Grid dimensions */
    fprintf(fp, "%5d%5d%5d\n",
            geometry.isize, geometry.jsize, geometry.ksize);
    
    /* Boundaries in x-, y- and z-directions */
    for (int ix = 0; ix<=geometry.isize; ix++) {
        fprintf(fp, "%f ", geometry.xbounds[ix]);
    }
    fprintf(fp, "\n");
    for (int iy = 0; iy<=geometry.jsize; iy++) {
        fprintf(fp, "%f ", geometry.ybounds[iy]);
    }
    fprintf(fp, "\n");
    for (int iz = 0; iz<=geometry.ksize; iz++) {
        fprintf(fp, "%f ", geometry.zbounds[iz]);
    }
    fprintf(fp, "\n");
    
    /* Dose or energy array */
    for (int iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                fprintf(fp, "%e ", score.accum_endep[irl]);
            }
        }
    }
    fprintf(fp, "\n");
    
    /* Uncertainty array */
    for (int iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                fprintf(fp, "%f ", score.accum_endep2[irl]);
            }
        }
    }
    fprintf(fp, "\n");
    
    /* Cleaning */
    fclose(fp);
    free(file_name);

    return;
}

/******************************************************************************/
/* Region-by-region definitions */
void initRegions() {
    
    /* +1 : consider region surrounding phantom */
    int nreg = geometry.isize*geometry.jsize*geometry.ksize + 1;
    
    /* Allocate memory for region data */
    region.med = malloc(nreg*sizeof(int));
    region.rhof = malloc(nreg*sizeof(double));
    region.pcut = malloc(nreg*sizeof(double));
    region.ecut = malloc(nreg*sizeof(double));
    
    /* First get global energy cutoff parameters */
    char buffer[BUFFER_SIZE];
    if (getInputValue(buffer, "global ecut") != 1) {
        printf("Can not find 'global ecut' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    double ecut = atof(buffer);
    
    if (getInputValue(buffer, "global pcut") != 1) {
        printf("Can not find 'global pcut' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    double pcut = atof(buffer);
    
    /* Initialize transport parameters on each region. Region 0 is outside the
     geometry */
    region.med[0] = VACUUM;
    region.rhof[0] = 0.0;
    region.pcut[0] = 0.0;
    region.ecut[0] = 0.0;
    
    for (int i=1; i<nreg; i++) {
        
        /* -1 : EGS counts media from 1. Substract 1 to get medium index */
        int imed = geometry.med_indices[i - 1] - 1;
        region.med[i] = imed;
        
        if (imed == VACUUM) {
            region.rhof[0] = 0.0F;
            region.pcut[0] = 0.0F;
            region.ecut[0] = 0.0F;
        }
        else {
            if (geometry.med_densities[i - 1] == 0.0F) {
                region.rhof[i] = 1.0;
            }
            else {
                region.rhof[i] =
                    geometry.med_densities[i - 1]/pegs_data.rho[imed];
            }
            
            /* Check if global cut-off values are within PEGS data */
            if (pegs_data.ap[imed] <= pcut) {
                region.pcut[i] = pcut;
            } else {
                printf("Warning!, global pcut value is below PEGS's pcut value "
                       "%f for medium %d, using PEGS value.\n",
                       pegs_data.ap[imed], imed);
                region.pcut[i] = pegs_data.ap[imed];
            }
            if (pegs_data.ae[imed] <= ecut) {
                region.ecut[i] = ecut;
            } else {
                printf("Warning!, global pcut value is below PEGS's ecut value "
                       "%f for medium %d, using PEGS value.\n",
                       pegs_data.ae[imed], imed);
            }
        }
    }
    
    return;
}

void initHistory() {

    double rnno1;
    double rnno2;
    
    /* Initialize first particle of the stack from source data */
    stack.np = 0;
    stack.iq[stack.np] = source.charge;
    
    /* Get primary particle energy */
    double ein = 0.0;
    if (source.spectrum) {
        /* Sample initial energy from spectrum data */
        rnno1 = setRandom();
        rnno2 = setRandom();
        
        /* Sample bin number in order to select particle energy */
        int k = (int)fmin(source.deltak*rnno1, source.deltak - 1.0);
        ein = source.cdfinv1[k] + rnno2*source.cdfinv2[k];
    }
    else {
        /* Monoenergetic source */
        ein = source.energy;
    }
    
    /* Check if the particle is an electron, in such a case add electron
     rest mass energy */
    if (stack.iq[stack.np] != 0) {
        /* Electron or positron */
        stack.e[stack.np] = ein + RM;
    }
    else {
        /* Photon */
        stack.e[stack.np] = ein;
    }
    
    /* Accumulate sampled kinetic energy for fraction of deposited energy
     calculations */
    score.ensrc += ein;
           
    /* Set particle position. First obtain a random position in the rectangle
     defined by the collimator */
    double rxyz = 0.0;
    if (source.xsize == 0.0 || source.ysize == 0.0) {
        stack.x[stack.np] = source.xinl;
        stack.y[stack.np] = source.yinl;
        
        rxyz = sqrt(pow(source.ssd, 2.0) + pow(stack.x[stack.np], 2.0) +
                    pow(stack.y[stack.np], 2.0));
        
        /* Get direction along z-axis */
        stack.w[stack.np] = source.ssd/rxyz;
        
    } else {
        double fw;
        double rnno3;
        do { /* rejection sampling of the initial position */
            rnno3 = setRandom();
            stack.x[stack.np] = rnno3*source.xsize + source.xinl;
            rnno3 = setRandom();
            stack.y[stack.np] = rnno3*source.ysize + source.yinl;
            rnno3 = setRandom();
            rxyz = sqrt(source.ssd*source.ssd + 
				stack.x[stack.np]*stack.x[stack.np] +
				stack.y[stack.np]*stack.y[stack.np]);
            
            /* Get direction along z-axis */
            stack.w[stack.np] = source.ssd/rxyz;
            fw = stack.w[stack.np]*stack.w[stack.np]*stack.w[stack.np];
        } while(rnno3 >= fw);
    }
    /* Set position of the particle in front of the geometry */
    stack.z[stack.np] = geometry.zbounds[0];
    
    /* At this point the position has been found, calculate particle
     direction */
    stack.u[stack.np] = stack.x[stack.np]/rxyz;
    stack.v[stack.np] = stack.y[stack.np]/rxyz;
    
    /* Determine region index of source particle */
    int ix, iy;
    if (source.xsize == 0.0) {
        ix = source.ixinl;
    } else {
        ix = source.ixinl - 1;
        while ((geometry.xbounds[ix+1] < stack.x[stack.np]) && ix < geometry.isize-1) {
            ix++;
        }
    }
    if (source.ysize == 0.0) {
        iy = source.iyinl;
    } else {
        iy = source.iyinl - 1;
        while ((geometry.ybounds[iy+1] < stack.y[stack.np]) && iy < geometry.jsize-1) {
            iy++;
        }
    }
    stack.ir[stack.np] = 1 + ix + iy*geometry.isize;
    
    /* Set statistical weight and distance to closest boundary*/
    stack.wt[stack.np] = 1.0;
    stack.dnear[stack.np] = 0.0;
        
    return;
}

/******************************************************************************/
/* omc_dosxyz main function */
int main (int argc, char **argv) {
    
    /* Execution time measurement */
    double tbegin;
    tbegin = omc_get_time();
    
    /* Parsing program options */
    
    int c;
    char *input_file = NULL;
    char *output_file = NULL;
    
    while (1) {
        static struct option long_options[] =
        {
            /* These options set a flag. */
            {"verbose", no_argument, &verbose_flag, 1},
            {"brief",   no_argument, &verbose_flag, 0},
            /* These options don’t set a flag.
             We distinguish them by their indices. */
            {"input",  required_argument, 0, 'i'},
            {"output",    required_argument, 0, 'o'},
            {0, 0, 0, 0}
        };
        
        /* getopt_long stores the option index here. */
        int option_index = 0;
        
        c = getopt_long(argc, argv, "i:o:",
                         long_options, &option_index);
        
        /* Detect the end of the options. */
        if (c == -1)
            break;
        
        switch (c) {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if (long_options[option_index].flag != 0)
                    break;
                printf ("option %s", long_options[option_index].name);
                if (optarg)
                    printf (" with arg %s", optarg);
                printf ("\n");
                break;
                
            case 'i':
                input_file = malloc(strlen(optarg) + 1);
                strcpy(input_file, optarg);
                printf ("option -i with value `%s'\n", input_file);
                break;
                
            case 'o':
                output_file = malloc(strlen(optarg) + 1);
                strcpy(output_file, optarg);
                printf ("option -o with value `%s'\n", output_file);
                break;
            
            case '?':
                /* getopt_long already printed an error message. */
                break;
                
            default:
                exit(EXIT_FAILURE);
        }
    }
    
    /* Instead of reporting ‘--verbose’
     and ‘--brief’ as they are encountered,
     we report the final status resulting from them. */
    if (verbose_flag)
        puts ("verbose flag is set");
    
    /* Print any remaining command line arguments (not options). */
    if (optind < argc)
    {
        printf ("non-option ARGV-elements: ");
        while (optind < argc)
            printf ("%s ", argv[optind++]);
        putchar ('\n');
    }
    
    /* Parse input file and print key,value pairs (test) */
    parseInputFile(input_file);

    /* Get information of OpenMP environment */
#ifdef _OPENMP
    int omp_size = omp_get_num_procs();
    printf("Number of OpenMP threads: %d\n", omp_size);
    omp_set_num_threads(omp_size);
#else
    printf("ompMC compiled without OpenMP support. Serial execution.\n");
#endif
    
    /* Read geometry information from phantom file and initialize geometry */
    initPhantom();
    
    /* With number of media and media names initialize the medium data */
    initMediaData();
    
    /* Initialize radiation source */
    initSource();
    
    /* Initialize data on a region-by-region basis */
    initRegions();

    /* Initialize VRT data */
    initVrt();
    
    /* Preparation of scoring struct */
    initScore();

    #pragma omp parallel
    {
      /* Initialize random number generator */
      initRandom();

      /* Initialize particle stack */
      initStack();
    }


    /* In verbose mode, list interaction data to output folder */
    if (verbose_flag) {
        listRayleigh();
        listPair();
        listPhoton();
        listElectron();
        listMscat();
        listSpin();
    }
    
    /* Shower call */
    
    /* Get number of histories and statistical batches */
    char buffer[BUFFER_SIZE];
    if (getInputValue(buffer, "ncase") != 1) {
        printf("Can not find 'ncase' key on input file.\n");
        exit(EXIT_FAILURE);
    }
   int nhist = atoi(buffer);
    
    if (getInputValue(buffer, "nbatch") != 1) {
        printf("Can not find 'nbatch' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    int nbatch = atoi(buffer);
    
    if (nhist/nbatch == 0) {
        nhist = nbatch;
    }
    
    int nperbatch = nhist/nbatch;
    nhist = nperbatch*nbatch;
    
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    printf("Total number of particle histories: %d\n", nhist);
    printf("Number of statistical batches: %d\n", nbatch);
    printf("Histories per batch: %d\n", nperbatch);
    
    /* Execution time up to this point */
    printf("Execution time up to this point : %8.2f seconds\n",
           (omc_get_time() - tbegin));
    
    for (int ibatch=0; ibatch<nbatch; ibatch++) {
        if (ibatch == 0) {
            /* Print header for information during simulation */
            printf("%-10s\t%-15s\t%-10s\n", "Batch #", "Elapsed time",
                   "RNG state");
            printf("%-10d\t%-15.2f\t%-5d%-5d\n", ibatch,
                   (omc_get_time() - tbegin), rng.ixx, rng.jxx);
        }
        else {
            /* Print state of current batch */
            printf("%-10d\t%-15.2f\t%-5d%-5d\n", ibatch,
                   (omc_get_time() - tbegin), rng.ixx, rng.jxx);
            
        }
        int ihist;
        #pragma omp parallel for schedule(dynamic)
        for (ihist=0; ihist<nperbatch; ihist++) {
            /* Initialize particle history */
            initHistory();
            
            /* Start electromagnetic shower simulation */
            shower();
        }
        
        /* Accumulate results of current batch for statistical analysis */
        accumEndep();
    }
    
    /* Print some output and execution time up to this point */
    printf("Simulation finished\n");
    printf("Execution time up to this point : %8.2f seconds\n",
           (omc_get_time() - tbegin));
    
    /* Analysis and output of results */
    if (verbose_flag) {
        /* Sum energy deposition in the phantom */
        double etot = 0.0;
        for (int irl=1; irl<gridsize+1; irl++) {
            etot += score.accum_endep[irl];
        }
        printf("Fraction of incident energy deposited in the phantom: %5.4f\n",
               etot/score.ensrc);
    }
    
    int iout = 1;   /* i.e. deposit mean dose per particle fluence */
    outputResults(output_file, iout, nperbatch, nbatch);
    
    /* Cleaning */
    cleanPhantom();
    cleanPhoton();
    cleanRayleigh();
    cleanPair();
    cleanElectron();
    cleanMscat();
    cleanSpin();
    cleanRegions();
    cleanScore();
    cleanSource();
    #pragma omp parallel
    {
      cleanRandom();
      cleanStack();
    }
    free(input_file);
    free(output_file);
    /* Get total execution time */
    printf("Total execution time : %8.5f seconds\n",
           (omc_get_time() - tbegin));
    
    exit (EXIT_SUCCESS);
}
