/******************************************************************************
 ompMC - An hybrid parallel implementation for Monte Carlo particle transport
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

#include "ompmc.h"
#include <stdio.h>
#include <stdlib.h>

/******************************************************************************/
/* A simple C/C++ class to parse input files and return requested
 key value -- https://github.com/bmaynard/iniReader */

#include <string.h>
#include <ctype.h>

/* Parse a configuration file */
void parseInputFile(char *input_file) {
    
    char buf[BUFFER_SIZE];      // support lines up to 120 characters
    
    /* Make space for the new string */
    char *extension = INPUT_EXT;
    char *file_name = malloc(strlen(input_file) + strlen(extension) + 1);
    strcpy(file_name, input_file);
    strcat(file_name, extension); /* add the extension */
    
    FILE *fp;
    if ((fp = fopen(file_name, "r")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
    while (fgets(buf, BUFFER_SIZE , fp) != NULL) {
        /* Jumps lines labeled with #, together with only white
         spaced or empty ones. */
        if (strstr(buf, "#") || lineBlack(buf)) {
            continue;
        }
        
        strcpy(input_items[input_idx].key, strtok(buf, "=\r\n"));
        strcpy(input_items[input_idx].value, strtok(NULL, "\r\n"));
        input_idx++;
    }
    
    input_idx--;
    fclose(fp);
    
    if(verbose_flag) {
        for (int i = 0; i<input_idx; i++) {
            printf("key = %s, value = %s\n", input_items[i].key,
                   input_items[i].value);
        }
    }

    /* Cleaning */
    free(file_name);
    
    return;
}

/* Copy the value of the selected input item to the char pointer */
int getInputValue(char *dest, char *key) {
    
    /* Check to see if anything got parsed */
    if (input_idx == 0) {
        return 0;
    }
    
    for (int i = 0; i <= input_idx; i++) {
        if (strstr(input_items[i].key, key)) {
            strcpy(dest, input_items[i].value);
            return 1;
        }
    }
    
    return 0;
}

/* Returns nonzero if line is a string containing only whitespace or is empty */
int lineBlack(char *line) {
    char * ch;
    int is_blank = 1;
    
    /* Iterate through each character. */
    for (ch = line; *ch != '\0'; ++ch) {
        if (!isspace(*ch)) {
            /* Found a non-whitespace character. */
            is_blank = 0;
            break;
        }
    }
    
    return is_blank;
}

/* Remove white spaces from string str_untrimmed and saves the results in
 str_trimmed. Useful for string input values, such as file names */
 void removeSpaces(char* str_trimmed,
                  const char* str_untrimmed) {
    
    while (*str_untrimmed != '\0') {
        if(!isspace(*str_untrimmed)) {
            *str_trimmed = *str_untrimmed;
            str_trimmed++;
        }
        str_untrimmed++;
    }
    
    *str_trimmed = '\0';
    return;
}

struct inputItems input_items[INPUT_PAIRS];     // key,value pairs
int input_idx = 0;                              // number of key,value pair

/******************************************************************************/

/*******************************************************************************
* Implementation, based on the EGSnrc one, of the RANMAR random number 
* generator (RNG), proposed by Marsaglia and Zaman. 
* 
* Following the EGSnrc implementation, it uses integers to store the state of 
* the RNG and to generate the next number in the sequence. Only at the end the 
* random numbers are converted to reals, due to performance reasons. 
* 
* Before using the RNG, it is needed to initialize the RNG by a call to 
* initRandom(). 
*******************************************************************************/

/* Initialization function for the RANMAR random number generator (RNG) 
proposed by Marsaglia and Zaman and adapted from the EGSnrc version to be 
used in ompMC. */
void initRandom() {
    
    /* Get initial seeds from input */
    char buffer[BUFFER_SIZE];
    if (getInputValue(buffer, "rng seeds") != 1) {
        printf("Can not find 'rng seeds' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    sscanf(buffer, "%d %d", &rng.ixx, &rng.jxx);
    
    if (rng.ixx <= 0 || rng.ixx > 31328) {
        printf("Warning!, setting Marsaglia default for ixx\n");
        rng.ixx = 1802; /* sets Marsaglia default */
    }
    if (rng.jxx <= 0 || rng.jxx > 31328) {
        printf("Warning!, setting Marsaglia default for jxx\n");
        rng.jxx = 9373; /* sets Marsaglia default */
    }
    printf("RNG seeds : ixx = %d, jxx = %d\n", rng.ixx, rng.jxx);
    
    int i = (rng.ixx/177 % 177) + 2;
    int j = (rng.ixx % 177) + 2;
    int k = (rng.jxx/169 % 178) + 1;
    int l = (rng.jxx % 169);
    
    int s, t, m;
    rng.urndm = malloc(97*sizeof(int));
    for (int ii = 0; ii<97; ii++) {
        s = 0;
        t = 8388608;    /* t is 2^23, half of the maximum allowed. Note that
                         only 24 bits are used */
        for (int jj=0; jj<24; jj++) {
            m = ((i*j % 179)*k) % 179;
            i = j;
            j = k;
            k = m;
            l = (53*l + 1) % 169;
            
            if (l*m % 64 >= 32) {
                s += t;
            }
            t /=2;
        }
        rng.urndm[ii] = s;
    }
    
    rng.crndm = 362436;
    rng.cdrndm = 7654321;
    rng.cmrndm = 16777213;
    
    rng.twom24 = 1.0/16777216.0;
    
    rng.ixx = 97;
    rng.jxx = 33;
    
    /* Allocate memory for random array and set seed to start calculation of
     random numbers */
    rng.rng_array = malloc(NRANDOM*sizeof(int));
    rng.rng_seed = NRANDOM;
    
    /* Print some information for debugging purposes */
    if (verbose_flag) {
        printf("Listing RNG information:\n");
        printf("ixx = %d\n", rng.ixx);
        printf("jxx = %d\n", rng.jxx);
        printf("crndm = %d\n", rng.crndm);
        printf("cdrndm = %d\n", rng.cdrndm);
        printf("cmrndm = %d\n", rng.cmrndm);
        printf("twom24 = %e\n", rng.twom24);
        printf("rng_seed = %d\n", rng.rng_seed);
        
        printf("\n");
        printf("urndm = \n:");
        for (int i=0; i<5; i++) { /* printf just 5 first values */
            printf("urndm[%d] = %d\n", i, rng.urndm[i]);
        }
        printf("\n");
    }
    return;
}

/* Generation function for the RANMAR random number generator (RNG) proposed 
by Marsaglia and Zaman. It generates NRANDOM floating point numbers in 
each call */
void getRandom() {
    
    int iopt;
    
    for (int i=0; i<NRANDOM; i++) {
        iopt = rng.urndm[rng.ixx - 1] - rng.urndm[rng.jxx - 1]; /* C index */
        if (iopt < 0) {
            iopt += 16777216;
        }
        
        rng.urndm[rng.ixx - 1] = iopt;

        rng.ixx -= 1;
        rng.jxx -= 1;
        if (rng.ixx == 0) {
            rng.ixx = 97;
        }
        else if (rng.jxx == 0) {
            rng.jxx = 97;
        }
        
        rng.crndm -= rng.cdrndm;
        if (rng.crndm < 0) {
            rng.crndm += rng.cmrndm;
        }
        
        iopt -= rng.crndm;
        if (iopt < 0) {
            iopt += 16777216;
        }
        rng.rng_array[i] = iopt;
    }
    
    rng.rng_seed = 0; /* index in C starts at 0 */
    
    return;
}

/* Get a single floating random number in [0,1) using the RANMAR RNG */
double setRandom() {
    
    double rnno = 0.0;
    
    if (rng.rng_seed >= NRANDOM) {
        getRandom();
    }
    
    rnno = rng.rng_array[rng.rng_seed]*rng.twom24;
    rng.rng_seed += 1;
    
    return rnno;
}

void cleanRandom() {
    
    free(rng.urndm);
    free(rng.rng_array);
    
    return;
}

/******************************************************************************/

/*******************************************************************************
/* Definitions for Monte Carlo simulation of particle transport 
*******************************************************************************/

/* Common functions and definitions */
struct Stack stack;

void initStack() {
    
    /* Allocate memory for particle stack */
    stack.np = 0;
    stack.iq = malloc(MXSTACK*sizeof(int));
    stack.ir = malloc(MXSTACK*sizeof(int));
    stack.e = malloc(MXSTACK*sizeof(double));
    stack.x = malloc(MXSTACK*sizeof(double));
    stack.y = malloc(MXSTACK*sizeof(double));
    stack.z = malloc(MXSTACK*sizeof(double));
    stack.u = malloc(MXSTACK*sizeof(double));
    stack.v = malloc(MXSTACK*sizeof(double));
    stack.w = malloc(MXSTACK*sizeof(double));
    stack.wt = malloc(MXSTACK*sizeof(double));
    stack.dnear = malloc(MXSTACK*sizeof(double));
    
    return;
}

void cleanStack() {
    
    free(stack.iq);
    free(stack.ir);
    free(stack.e);
    free(stack.x);
    free(stack.y);
    free(stack.z);
    free(stack.u);
    free(stack.v);
    free(stack.w);
    free(stack.wt);
    free(stack.dnear);
    
    return;
}

void transferProperties(int npnew, int npold) {
    /* The following function transfer phase space properties from particle
     npold on stack to particle np */
    stack.x[npnew] = stack.x[npold];
    stack.y[npnew] = stack.y[npold];
    stack.z[npnew] = stack.z[npold];

    stack.u[npnew] = stack.u[npold];
    stack.v[npnew] = stack.v[npold];
    stack.w[npnew] = stack.w[npold];

    stack.ir[npnew] = stack.ir[npold];
    stack.wt[npnew] = stack.wt[npold];
    stack.dnear[npnew] = stack.dnear[npold];
    
    return;
}

void selectAzimuthalAngle(double *costhe, double *sinthe) {
    /* Function for azimuthal angle selecton using a sampling within a box 
    method */
    double xphi, xphi2, yphi, yphi2, rhophi2;

    do {
        xphi = setRandom();
        xphi = 2.0*xphi - 1.0;
        xphi2 = xphi*xphi;

        yphi = setRandom();
        yphi2  = yphi*yphi;
        rhophi2 = xphi2 + yphi2;        
    } while(rhophi2 > 1.0);

    rhophi2 = 1/rhophi2;
    *costhe = (xphi2 - yphi2)*rhophi2;
    *sinthe = 2.0*xphi*yphi*rhophi2;

    return;
}

/* The following set of uphi functions set coordinates for new particle or
reset direction cosines of old one. Generate random azimuth selection and
replace the direction cosine with their new values. */

void uphi21(struct Uphi *uphi,
            double costhe, double sinthe) {

    int np = stack.np;

    /* This section is used if costhe and sinthe are already known. Phi
    is selected uniformly over the interval (0,2Pi) */
    selectAzimuthalAngle(&(uphi->cosphi), &(uphi->sinphi));
    
    /* The following section is used for the second of two particles when it is
    known that there is a relationship in their corrections. In this version
    it is worked on the old particle */
    uphi->A = stack.u[np];
    uphi->B = stack.v[np];
    uphi->C = stack.w[np];
    
    double sinps2 = pow(uphi->A, 2.0) + pow(uphi->B, 2.0);
    
    /* Small polar change */
    if (sinps2 < 1.0E-20) {
        stack.u[np] = sinthe*uphi->cosphi;
        stack.v[np] = sinthe*uphi->sinphi;
        stack.w[np] = uphi->C*costhe;
    }
    else {
        double sinpsi = sqrt(sinps2);
        double us = sinthe*uphi->cosphi;
        double vs = sinthe*uphi->sinphi;
        double sindel = uphi->B/sinpsi;
        double cosdel = uphi->A/sinpsi;
        
        stack.u[np] = uphi->C*cosdel*us - sindel*vs + uphi->A*costhe;
        stack.v[np] = uphi->C*sindel*us + cosdel*vs + uphi->B*costhe;
        stack.w[np] = -sinpsi*us + uphi->C*costhe;
    }
    
    return;
}

void uphi32(struct Uphi *uphi,
            double costhe, double sinthe) {
    
    int np = stack.np;
    
    /* The following section is used for the second of two particles when it is
    known that there is a relationship in their corrections. In this version
    it is worked on the new particle */
    
    /* Transfer phase space information like position and direction of the
    first particle to the second */
    transferProperties(np, np-1);
    
    /* Now adjust direction of the second particle */
    double sinps2 = pow(uphi->A, 2.0) + pow(uphi->B, 2.0);
    
    /* Small polar change */
    if (sinps2 < 1E-20) {
        stack.u[np] = sinthe*uphi->cosphi;
        stack.v[np] = sinthe*uphi->sinphi;
        stack.w[np] = uphi->C*costhe;
    }
    else {
        double sinpsi = sqrt(sinps2);
        double us = sinthe*uphi->cosphi;
        double vs = sinthe*uphi->sinphi;
        double sindel = uphi->B/sinpsi;
        double cosdel = uphi->A/sinpsi;
        
        stack.u[np] = uphi->C*cosdel*us - sindel*vs + uphi->A*costhe;
        stack.v[np] = uphi->C*sindel*us + cosdel*vs + uphi->B*costhe;
        stack.w[np] = -sinpsi*us + uphi->C*costhe;
    }
    
    return;
}

/*******************************************************************************
/* Photon physical processes definitions
*******************************************************************************/

struct Photon photon_data;

void readXsecData(char *file, int *ndat,
                  double **xsec_data0,
                  double **xsec_data1) {
    
    /* Open cross-section file */
    FILE *fp;
    
    if ((fp = fopen(file, "r")) == NULL) {
        printf("Unable to open file: %s\n", file);
        exit(EXIT_FAILURE);
    }
    
    printf("Path to cross-section file : %s\n", file);
    
    int ok = fp > 0; // "boolean" variable, ok = 0, false; ok = 1, true
    
    if(ok == 1) {
        
        printf("Reading cross-section data file: %s\n", file);
        
        for (int i=0; i<MXELEMENT; i++) {
            
            int n;
            
            if (fscanf(fp, "%u\n", &n) != 1) {
                ok = 0;
                break;
            }
            
            ndat[i] = n;
            xsec_data0[i] = malloc(n*sizeof(double));
            xsec_data1[i] = malloc(n*sizeof(double));
            
            for (int j=0; j<n; j++) {
                
                double dat0, dat1;
                
                if (fscanf(fp, "%lf %lf", &dat0, &dat1) != 2) {
                    ok = 0;
                    break;
                }
                
                xsec_data0[i][j] = dat0;
                xsec_data1[i][j] = dat1;
            }
            
            if (ok == 0) {
                break;
            }
        }
    }
    
    if (fp) {
        fclose(fp);
    }
    
    if (ok == 0) {
        printf("Could not read the data file %s\n", file);
        exit(EXIT_FAILURE);
    }

    return;
}

void heap_sort(int n, double *values, int *indices) {
    /* Sort the array values and at the same time changes the corresponding
     array of indices */
    for (int i = 0; i < n; i++) {
        indices[i] = i + 1;
    }
    
    if (n < 2) {
        return;
    }
    
    int l = n/2 + 1;
    int idx = n;
    
    int i, j;
    double last_value;
    int last_idx;
    
    do {
        if (l > 1) {
            l--;
            last_value = values[l-1];
            last_idx = l;
        }
        else {
            last_value = values[idx-1];
            last_idx = indices[idx-1];
            values[idx-1] = values[0];
            indices[idx-1] = indices[0];
            idx--;
            if (idx == 0) {
                values[0] = last_value;
                indices[0] = last_idx;
                return;
            }
        }
        
        i = l;
        j = 2*l;
        
        do {
            if (j > idx) {
                break;
            }
            
            if (j < idx) {
                if (values[j-1] < values[j]) {
                    j++;
                }
            }
            if (last_value < values[j-1]) {
                values[i-1] = values[j-1];
                indices[i-1] = indices[j-1];
                i = j;
                j = 2*j;
            }
            else
                j = idx + 1;
        } while(1);
        
        values[i-1] = last_value;
        indices[i-1] = last_idx;
    } while(1);
    
    return;
}

double *get_data(int flag,
                 int ne,
                 int *ndat,
                 double **data0,
                 double **data1,
                 double *z_sorted,
                 double *pz_sorted,
                 double ge0, double ge1) {
    
    /* Allocate space for the result returned by get_data() */
    double *res = (double*)malloc(MXGE * sizeof(double));
    
    for (int i=0; i<MXGE; i++) {
        res[i] = 0.0;
    }
    
    for (int i=0; i<ne; i++) {
        int z = (int)(z_sorted[i] + 0.5)-1;
        int n = ndat[z];
        double eth = 0.0;
        
        double *in_dat0;
        double *in_dat1;
        
        if (flag == 0) {
            in_dat0 = malloc(n*sizeof(double));
            in_dat1 = malloc(n*sizeof(double));
            
            for (int j=0; j<n; j++) {
                in_dat0[j] = data0[z][j];
                in_dat1[j] = data1[z][j];
            }
        }
        else {
            in_dat0 = malloc((n+1)*sizeof(double));
            in_dat1 = malloc((n+1)*sizeof(double));
            
            for (int j=0; j<n; j++) {
                in_dat0[j + 1] = data0[z][j];
                in_dat1[j + 1] = data1[z][j];
            }
            
            if (flag == 1) {
                eth = 2.0*RM;
            }
            else {
                eth = 4.0*RM;
            }
            
            n++;
            
            for (int j=1; j<n; j++) {
                in_dat1[j] -= 3.0*log(1.0-eth/exp(in_dat0[j]));
            }
            
            in_dat0[0] = (double)log(eth);
            in_dat1[0] = (double)in_dat1[1];
        }
        
        for (int j=0; j<MXGE; j++) {
            /* Added +1 to j below due to C loop starting at 0 */
            double gle = ((double)(j+1) - ge0) / ge1;
            double e = exp(gle);
            double sig = 0.0;
            
            if ((gle < in_dat0[0]) || (gle >= in_dat0[n-1])) {
                if (flag == 0) {
                    printf(" Energy %f is outside the available data range of "
                           "%f to %f.\n", e, exp(in_dat0[0]),
                           exp(in_dat0[n-1]));
                }
                else {
                    if (gle < in_dat0[0]) {
                        sig = 0.0;
                    }
                    else {
                        sig = exp(in_dat1[n-1]);
                    }
                }
            }
            else {
                int k;
                for (k=0; k<n-1; k++) {
                    if ((gle >= in_dat0[k]) && (gle < in_dat0[k+1])) {
                        break;
                    }
                }
                double p = (gle - in_dat0[k])/(in_dat0[k+1] - in_dat0[k]);
                sig = exp(p*in_dat1[k+1] + (1.0 - p)*in_dat1[k]);
            }
            if ((flag != 0) && (e > eth)) {
                sig *= (1.0 - eth/e)*(1.0 - eth/e)*(1.0 - eth/e);
            }
            
            res[j] += pz_sorted[i]*sig;
        }
        
        free(in_dat0);
        free(in_dat1);
    }
    
    return res;
}

double kn_sigma0(double e) {
    /* Compton cross-section calculation */
    
    double con = 0.1274783851;
    double ko = e/RM;
    
    if (ko < 0.01) {
        return 8.0*con/3.0*(1.0-ko*(2.0-ko*(5.2-13.3*ko)))/RM;
    }
    
    double c1 = 1.0/(ko*ko);
    double c2 = 1.0 - 2.0*(1.0 + ko)*c1;
    double c3 = (1.0 + 2.0*ko)*c1;
    double eps2 = 1.0;
    double eps1 = 1.0 / (1.0 + 2.0*ko);
    
    return (c1*(1.0/eps1 - 1.0/eps2) + c2*log(eps2/eps1) +
            eps2*(c3 + 0.5*eps2) - eps1*(c3 + 0.5*eps1))/e*con;
}

void initPhotonData() {
    
    /* Get file path from input data */
    char photon_xsection[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "data folder") != 1) {
        printf("Can not find 'data folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(photon_xsection, buffer);
    
    /* Get specific cross-section data and saves it in
     'interaction'_xsec_data array */
    int *photo_ndat = (int*) malloc(MXELEMENT*sizeof(int));
    double **photo_xsec_data0 = (double**) malloc(MXELEMENT*sizeof(double*));
    double **photo_xsec_data1 = (double**) malloc(MXELEMENT*sizeof(double*));
    
    char xsection_file[256];
    strcpy(xsection_file, photon_xsection);
    strcat(xsection_file, "xcom_photo.data");
    readXsecData(xsection_file, photo_ndat, photo_xsec_data0, photo_xsec_data1);
    
    int *rayleigh_ndat = (int*) malloc(MXELEMENT*sizeof(int));
    double **rayleigh_xsec_data0 = (double**) malloc(MXELEMENT*sizeof(double*));
    double **rayleigh_xsec_data1 = (double**) malloc(MXELEMENT*sizeof(double*));
    
    strcpy(xsection_file, photon_xsection);
    strcat(xsection_file, "xcom_rayleigh.data");
    readXsecData(xsection_file, rayleigh_ndat, rayleigh_xsec_data0,
                 rayleigh_xsec_data1);
    
    int *pair_ndat = (int*) malloc(MXELEMENT*sizeof(int));
    double **pair_xsec_data0 = (double**) malloc(MXELEMENT*sizeof(double*));
    double **pair_xsec_data1 = (double**) malloc(MXELEMENT*sizeof(double*));
    
    strcpy(xsection_file, photon_xsection);
    strcat(xsection_file, "xcom_pair.data");
    readXsecData(xsection_file, pair_ndat, pair_xsec_data0, pair_xsec_data1);
    
    /* We do not consider bound compton scattering, therefore there is no
     cross sections needed for compton scattering */
    
    int *triplet_ndat = (int*) malloc(MXELEMENT*sizeof(int));
    double **triplet_xsec_data0 = (double**) malloc(MXELEMENT*sizeof(double*));
    double **triplet_xsec_data1 = (double**) malloc(MXELEMENT*sizeof(double*));
    
    strcpy(xsection_file, photon_xsection);
    strcat(xsection_file, "xcom_triplet.data");
    readXsecData(xsection_file, triplet_ndat, triplet_xsec_data0,
                 triplet_xsec_data1);
    
    /* binding energies per element removed, as it is not currently supported */
    
    photon_data.ge0 = malloc(media.nmed*sizeof(double));
    photon_data.ge1 = malloc(media.nmed*sizeof(double));
    photon_data.gmfp0 = malloc(media.nmed*MXGE*sizeof(double));
    photon_data.gmfp1 = malloc(media.nmed*MXGE*sizeof(double));
    photon_data.gbr10 = malloc(media.nmed*MXGE*sizeof(double));
    photon_data.gbr11 = malloc(media.nmed*MXGE*sizeof(double));
    photon_data.gbr20 = malloc(media.nmed*MXGE*sizeof(double));
    photon_data.gbr21 = malloc(media.nmed*MXGE*sizeof(double));
    photon_data.cohe0 = malloc(media.nmed*MXGE*sizeof(double));
    photon_data.cohe1 = malloc(media.nmed*MXGE*sizeof(double));
    
    for (int i=0; i<media.nmed; i++) {
        photon_data.ge1[i] = (double)(MXGE - 1)/log(pegs_data.up[i]/pegs_data.ap[i]);
        photon_data.ge0[i] = 1.0 - photon_data.ge1[i]*log(pegs_data.ap[i]);
        
        double sumA = 0.0;
        double sumZ = 0.0;
        double *z_sorted = (double*) malloc(pegs_data.ne[i]*sizeof(double));
        
        for (int j=0; j<pegs_data.ne[i]; j++) {
            z_sorted[j] = pegs_data.elements[i][j].z;
            sumA += pegs_data.elements[i][j].pz*pegs_data.elements[i][j].wa;
            sumZ += pegs_data.elements[i][j].pz*pegs_data.elements[i][j].z;
        }
        double con2 = pegs_data.rho[i]/(sumA*1.6605655);
        int *sorted = (int*) malloc(pegs_data.ne[i]*sizeof(int));
        
        heap_sort(pegs_data.ne[i], z_sorted, sorted);
        
        double *pz_sorted = (double*)malloc(pegs_data.ne[i]*sizeof(double));
        for (int j = 0; j < pegs_data.ne[i]; j++) {
            pz_sorted[j] =pegs_data.elements[i][sorted[j]-1].pz; // C indexing
        }
        
        double *sig_photo = get_data(0, pegs_data.ne[i], photo_ndat,
                                     photo_xsec_data0, photo_xsec_data1,
                                     z_sorted, pz_sorted,
                                     photon_data.ge0[i], photon_data.ge1[i]);
        double *sig_rayleigh = get_data(0, pegs_data.ne[i], rayleigh_ndat,
                                        rayleigh_xsec_data0, rayleigh_xsec_data1
                                        , z_sorted, pz_sorted,
                                        photon_data.ge0[i], photon_data.ge1[i]);
        double *sig_pair = get_data(1, pegs_data.ne[i], pair_ndat,
                                    pair_xsec_data0, pair_xsec_data1,
                                    z_sorted, pz_sorted,
                                    photon_data.ge0[i], photon_data.ge1[i]);
        double *sig_triplet = get_data(2, pegs_data.ne[i], triplet_ndat,
                                       triplet_xsec_data0, triplet_xsec_data1,
                                       z_sorted, pz_sorted,
                                       photon_data.ge0[i], photon_data.ge1[i]);
        
        double gle = 0.0, gmfp = 0.0, gbr1 = 0.0, gbr2 = 0.0, cohe = 0.0;
        double gmfp_old = 0.0, gbr1_old = 0.0, gbr2_old = 0.0,
        cohe_old = 0.0;
        
        for (int j=0; j<MXGE; j++) {
            /* Added +1 to j below due to C loop starting at 0 */
            gle = ((double)(j+1) - photon_data.ge0[i]) / photon_data.ge1[i];
            double e = exp(gle);
            double sig_kn = sumZ*kn_sigma0(e);
            
            double sig_p = sig_pair[j] + sig_triplet[j];
            double sigma = sig_kn + sig_p + sig_photo[j];
            gmfp = 1.0/(sigma * con2);
            gbr1 = sig_p/sigma;
            gbr2 = gbr1 + sig_kn/sigma;
            cohe = sigma/(sig_rayleigh[j] + sigma);
            
            if (j > 0) {
                int idx = i*MXGE + (j-1); /* the -1 is not for C indexing! */
                photon_data.gmfp1[idx] = (gmfp - gmfp_old)*photon_data.ge1[i];
                photon_data.gmfp0[idx] = gmfp - photon_data.gmfp1[idx]*gle;
                
                photon_data.gbr11[idx] = (gbr1 - gbr1_old)*photon_data.ge1[i];
                photon_data.gbr10[idx] = gbr1 - photon_data.gbr11[idx]*gle;
                
                photon_data.gbr21[idx] = (gbr2 - gbr2_old)*photon_data.ge1[i];
                photon_data.gbr20[idx] = gbr2 - photon_data.gbr21[idx]*gle;
                
                photon_data.cohe1[idx] = (cohe - cohe_old)*photon_data.ge1[i];
                photon_data.cohe0[idx] = cohe - photon_data.cohe1[idx]*gle;
            }
            
            gmfp_old = gmfp;
            gbr1_old = gbr1;
            gbr2_old = gbr2;
            cohe_old = cohe;
        }
        
        int idx = i*MXGE + MXGE - 1;
        photon_data.gmfp1[idx] = photon_data.gmfp1[idx-1];
        photon_data.gmfp0[idx] = gmfp - photon_data.gmfp1[idx]*gle;
        
        photon_data.gbr11[idx] = photon_data.gbr11[idx-1];
        photon_data.gbr10[idx] = gbr1 - photon_data.gbr11[idx]*gle;
        
        photon_data.gbr21[idx] = photon_data.gbr21[idx-1];
        photon_data.gbr20[idx] = gbr2 - photon_data.gbr21[idx]*gle;
        
        photon_data.cohe1[idx] = photon_data.cohe1[idx-1];
        photon_data.cohe0[idx] = cohe - photon_data.cohe1[idx]*gle;
        
        /* Cleaning */
        free(z_sorted);
        free(sorted);
        free(pz_sorted);
        
        free(sig_photo);
        free(sig_rayleigh);
        free(sig_pair);
        free(sig_triplet);
    }
    
    /* Cleaning */
    free(photo_ndat);
    free(rayleigh_ndat);
    free(pair_ndat);
    free(triplet_ndat);
    
    for (int i=0; i<MXELEMENT; i++) {
        free(photo_xsec_data0[i]);
        free(photo_xsec_data1[i]);
        free(rayleigh_xsec_data0[i]);
        free(rayleigh_xsec_data1[i]);
        free(pair_xsec_data0[i]);
        free(pair_xsec_data1[i]);
        free(triplet_xsec_data0[i]);
        free(triplet_xsec_data1[i]);
    }
    
    free(photo_xsec_data0);
    free(photo_xsec_data1);
    free(rayleigh_xsec_data0);
    free(rayleigh_xsec_data1);
    free(pair_xsec_data0);
    free(pair_xsec_data1);
    free(triplet_xsec_data0);
    free(triplet_xsec_data1);
    
    return;
}

void cleanPhoton() {
    
    free(photon_data.ge0);
    free(photon_data.ge1);
    free(photon_data.gmfp0);
    free(photon_data.gmfp1);
    free(photon_data.gbr10);
    free(photon_data.gbr11);
    free(photon_data.gbr20);
    free(photon_data.gbr21);
    free(photon_data.cohe0);
    free(photon_data.cohe1);
    
    return;
}

void listPhoton() {

    /* Get file path from input data */
    char output_folder[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "output folder") != 1) {
        printf("Can not find 'output folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(output_folder, buffer);
    
    char file_name[256];
    strcpy(file_name, output_folder);
    strcat(file_name, "photon_data.lst");    

    /* List photon data to output file */
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "Listing photon data: \n");
    for (int i=0; i<media.nmed; i++) {
        fprintf(fp, "For medium %s: \n", media.med_names[i]);
        fprintf(fp, "photon_data.ge = \n");
        fprintf(fp, "\t ge0[%d] = %15.5f, ge1[%d] = %15.5f\n", i,
                photon_data.ge0[i], i, photon_data.ge1[i]);
        
        fprintf(fp, "photon_data.gmfp = \n");
        for (int j=0; j<MXGE; j++) {
            int idx = i*MXGE + j;
            fprintf(fp, "gmfp0[%d][%d] = %15.5f, gmfp1[%d][%d] = %15.5f\n",
                    j, i, photon_data.gmfp0[idx],
                    j, i, photon_data.gmfp1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "photon_data.gbr1 = \n");
        for (int j=0; j<MXGE; j++) {
            int idx = i*MXGE + j;
            fprintf(fp, "gbr10[%d][%d] = %15.5f, gbr11[%d][%d] = %15.5f\n",
                    j, i, photon_data.gbr10[idx],
                    j, i, photon_data.gbr11[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "photon_data.gbr2 = \n");
        for (int j=0; j<MXGE; j++) {
            int idx = i*MXGE + j;
            fprintf(fp, "gbr20[%d][%d] = %15.5f, gbr21[%d][%d] = %15.5f\n",
                    j, i, photon_data.gbr20[idx],
                    j, i, photon_data.gbr21[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "photon_data.cohe = \n");
        for (int j=0; j<MXGE; j++) {
            int idx = i*MXGE + j;
            fprintf(fp, "cohe0[%d][%d] = %15.5f, cohe1[%d][%d] = %15.5f\n",
                    j, i, photon_data.cohe0[idx],
                    j, i, photon_data.cohe1[idx]);
        }
        fprintf(fp, "\n");
        
    }
    
    fclose(fp);
    
    return;
}

/* Rayleigh scattering definitions */
struct Rayleigh rayleigh_data;

void readFfData(double *xval, double **aff) {
    
    /* Get file path from input data */
    char pgs4form_file[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "pgs4form file") != 1) {
        printf("Can not find 'pgs4form file' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(pgs4form_file, buffer);
    
    /* Open pgs4form file */
    FILE *fp;
    
    if ((fp = fopen(pgs4form_file, "r")) == NULL) {
        printf("Unable to open file: %s\n", pgs4form_file);
        exit(EXIT_FAILURE);
    }

    printf("Path to pgs4form file : %s\n", pgs4form_file);


    int ok = fp > 0; // "boolean" variable, ok = 0, false; ok = 1, true
    
    if (ok == 1) {
        /* Read momentum transfer values */
        for (int i=0; i<MXRAYFF; i++) {
            if (fscanf(fp, "%lf", &xval[i]) != 1) {
                ok = 0;
                break;
            }
        }
    }
    
    if (ok == 1) {
        /* Read element form factors */
        for (int i=0; i<MXELEMENT; i++) {
            for (int j=0; j<MXRAYFF; j++) {
                if (fscanf(fp, "%lf", &aff[i][j]) != 1) {
                    ok = 0;
                    break;
                }
            }
            if (ok == 0) {
                break;
            }
        }
    }
    
    if (fp) {
        fclose(fp);
    }
    
    if (ok == 0) {
        printf("Could not read atomic form factors file %s", 
                pgs4form_file);
        exit(EXIT_FAILURE);
    }
    
    return;
}

void initRayleighData(void) {
    
    double *xval = (double*) malloc(MXRAYFF*sizeof(double));
    double **aff = (double**) malloc(MXELEMENT*sizeof(double*));
    double *ff = malloc(MXRAYFF*media.nmed*sizeof(double));
    double *pe_array = malloc(MXGE*media.nmed*sizeof(double));

    for (int i=0; i<MXELEMENT; i++) {
        aff[i] = malloc(MXRAYFF*sizeof(double));
    }
    
    /* Read Rayleigh atomic form factor from pgs4form.dat file */
    readFfData(xval, aff);
    
    /* Allocate memory for Rayleigh data */
    rayleigh_data.xgrid = malloc(media.nmed*MXRAYFF*sizeof(double));
    rayleigh_data.fcum = malloc(media.nmed*MXRAYFF*sizeof(double));
    rayleigh_data.b_array = malloc(media.nmed*MXRAYFF*sizeof(double));
    rayleigh_data.c_array = malloc(media.nmed*MXRAYFF*sizeof(double));
    rayleigh_data.i_array = malloc(media.nmed*RAYCDFSIZE*sizeof(int));
    rayleigh_data.pmax0 = malloc(media.nmed*MXGE*sizeof(double));
    rayleigh_data.pmax1 = malloc(media.nmed*MXGE*sizeof(double));
    
    for (int i=0; i<media.nmed; i++) {
        /* Calculate form factor using independent atom model */
        for (int j=0; j<MXRAYFF; j++) {
            double ff_val = 0.0;
            rayleigh_data.xgrid[i*MXRAYFF + j] = xval[j];
            
            for (int k=0; k<pegs_data.ne[i]; k++) {
                int z = (int)pegs_data.elements[i][k].z - 1; /* C indexing */
                ff_val += pegs_data.elements[i][k].pz * pow(aff[z][j],2);
            }
            
            ff[i*MXRAYFF + j] = sqrt(ff_val);
        }
        
        if (rayleigh_data.xgrid[i*MXRAYFF] < 1.0E-6) {
            rayleigh_data.xgrid[i*MXRAYFF] = 0.0001;
        }
        
        /* Calculate rayleigh data, as in subroutine prepare_rayleigh_data
         inside EGSnrc*/
        double emin = exp((1.0 - photon_data.ge0[i])/photon_data.ge1[i]);
        double emax = exp((MXGE - photon_data.ge0[i])/photon_data.ge1[i]);
        
        /* The following is to avoid log(0) */
        for (int j=0; j<MXRAYFF; j++) {
            if (*((unsigned long*)&ff[i*MXRAYFF + j]) == 0) {
                unsigned long zero = 1;
                ff[i*MXRAYFF + j] = *((double*)&zero);
            }
        }
        
        /* Calculating the cumulative distribution */
        double sum0 = 0.0;
        rayleigh_data.fcum[i*MXRAYFF] = 0.0;
        
        for (int j=0; j < MXRAYFF-1; j++) {
            double b = log(ff[i*MXRAYFF + j + 1]
                           /ff[i*MXRAYFF + j])
                                /log(rayleigh_data.xgrid[i*MXRAYFF + j + 1]
                                /rayleigh_data.xgrid[i*MXRAYFF + j]);
            rayleigh_data.b_array[i*MXRAYFF + j] = b;
            double x1 = rayleigh_data.xgrid[i*MXRAYFF + j];
            double x2 = rayleigh_data.xgrid[i*MXRAYFF + j + 1];
            double pow_x1 = pow(x1, 2.0*b);
            double pow_x2 = pow(x2, 2.0*b);
            sum0 += pow(ff[i*MXRAYFF + j],2)
                *(pow(x2,2)*pow_x2 - pow(x1,2)*pow_x1)/((1.0 + b)*pow_x1);
            rayleigh_data.fcum[i*MXRAYFF + j + 1] = sum0;
        }
        
        /* Now the maximum cumulative propability as a function of incident
         photon energy */
        double dle = log(emax/emin)/((double)MXGE - 1.0);
        int idx = 1;
        
        for (int j=1; j<=MXGE; j++) {
            double e = emin*exp(dle*((double)j-1.0));
            double xmax = 20.607544*2.0*e/RM;
            int k;
            for (k=1; k<=MXRAYFF-1; k++) {
                if ((xmax >= rayleigh_data.xgrid[i*MXRAYFF + k - 1]) &&
                    (xmax < rayleigh_data.xgrid[i*MXRAYFF + k]))
                    break;
            }
            
            idx = k;
            double b = rayleigh_data.b_array[i*MXRAYFF + idx - 1];
            double x1 = rayleigh_data.xgrid[i*MXRAYFF + idx - 1];
            double x2 = xmax;
            double pow_x1 = pow(x1, 2.0 * b);
            double pow_x2 = pow(x2, 2.0 * b);
            pe_array[i*MXGE + j - 1] =
                rayleigh_data.fcum[i*MXRAYFF + idx - 1] +
                pow(ff[i * MXRAYFF + idx - 1], 2) *
                (pow(x2,2)*pow_x2 - pow(x1,2)*pow_x1)/((1.0 + b)*pow_x1);
            
        }
        
        rayleigh_data.i_array[i*RAYCDFSIZE + RAYCDFSIZE - 1] = idx;
        
        /* Now renormalize data so that pe_array(emax) = 1. Note that we make
         pe_array(j) slightly larger so that fcum(xmax) is never underestimated
         when interpolating */
        double anorm = 1.0/sqrt(pe_array[i*MXGE + MXGE - 1]);
        double anorm1 = 1.005/pe_array[i*MXGE + MXGE - 1];
        double anorm2 = 1.0/pe_array[i*MXGE + MXGE - 1];
        
        for (int j=0; j<MXGE; j++) {
            pe_array[i*MXGE + j] *= anorm1;
            if (pe_array[i*MXGE + j] > 1.0) {
                pe_array[i*MXGE + j] = 1.0;
            }
        }
        
        for (int j=0; j<MXRAYFF; j++) {
            ff[i*MXRAYFF + j] *= anorm;
            rayleigh_data.fcum[i*MXRAYFF + j] *= anorm2;
            rayleigh_data.c_array[i*MXRAYFF + j] = (1.0 +
                rayleigh_data.b_array[i*MXRAYFF + j])/
            pow(rayleigh_data.xgrid[i*MXRAYFF + j]*ff[i*MXRAYFF + j],2);
        }
        
        /* Now prepare uniform cumulative bins */
        double dw = 1.0/((double)RAYCDFSIZE - 1.0);
        double xold = rayleigh_data.xgrid[i*MXRAYFF + 0];
        int ibin = 1;
        double b = rayleigh_data.b_array[i*MXRAYFF + 0];
        double pow_x1 = pow(rayleigh_data.xgrid[i*MXRAYFF + 0], 2.0*b);
        rayleigh_data.i_array[i*MXRAYFF + 0] = 1;
        
        for (int j=2; j<=RAYCDFSIZE-1; j++) {
            double w = dw;
            
            do {
                double x1 = xold;
                double x2 = rayleigh_data.xgrid[i*MXRAYFF + ibin];
                double t = pow(x1, 2)*pow(x1, 2.0*b);
                double pow_x2 = pow(x2, 2.0*b);
                double aux = pow(ff[i*MXRAYFF + ibin - 1], 2)*
                    (pow(x2, 2)*pow_x2 - t)/((1.0 + b)*pow_x1);
                if (aux > w) {
                    xold = exp(log(t + w*(1.0 + b)*pow_x1/
                                   pow(ff[i*MXRAYFF + ibin - 1], 2))/
                               (2.0 + 2.0*b));
                    rayleigh_data.i_array[i*RAYCDFSIZE + j - 1] = ibin;
                    break;
                }
                w -= aux;
                xold = x2;
                ibin++;
                b = rayleigh_data.b_array[i*MXRAYFF + ibin - 1];
                pow_x1 = pow(xold, 2.0*b);
            } while (1);
        }
        
        /* change definition of b_array because that is what is needed at
         run time*/
        for (int j=0; j<MXRAYFF; j++) {
            rayleigh_data.b_array[i*MXRAYFF + j] = 0.5/(1.0 +
                rayleigh_data.b_array[i*MXRAYFF + j]);
        }
        
        /* Prepare coefficients for pmax interpolation */
        for (int j=0; j<MXGE-1; j++) {
            double gle = ((j+1) - photon_data.ge0[i])/photon_data.ge1[i];
            rayleigh_data.pmax1[i*MXGE + j] = (pe_array[i*MXGE + j + 1] -
                pe_array[i * MXGE + j])*photon_data.ge1[i];
            rayleigh_data.pmax0[i*MXGE + j] = pe_array[i*MXGE + j] -
                rayleigh_data.pmax1[i * MXGE + j]*gle;
        }
        rayleigh_data.pmax0[i*MXGE + MXGE - 1] = rayleigh_data.pmax0[i*MXGE + MXGE - 2];
        rayleigh_data.pmax1[i*MXGE + MXGE - 1] = rayleigh_data.pmax1[i*MXGE + MXGE - 2];
    }
    
    /* Cleaning */
    free(xval);
    free(ff);
    free(pe_array);
    for (int i=0; i<MXELEMENT; i++) {
        free(aff[i]);
    }
    free(aff);
    
    return;
}

void cleanRayleigh() {
    
    free(rayleigh_data.xgrid);
    free(rayleigh_data.b_array);
    free(rayleigh_data.c_array);
    free(rayleigh_data.fcum);
    free(rayleigh_data.i_array);
    free(rayleigh_data.pmax0);
    free(rayleigh_data.pmax1);
    
    return;
}

void listRayleigh() {
       
    /* Get file path from input data */
    char output_folder[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "output folder") != 1) {
        printf("Can not find 'output folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(output_folder, buffer);
    
    char file_name[256];
    strcpy(file_name, output_folder);
    strcat(file_name, "rayleigh_data.lst");
    
    /* List rayleigh data to output file */
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "Listing rayleigh data: \n");
    for (int i=0; i<media.nmed; i++) {
        fprintf(fp, "For medium %s: \n", media.med_names[i]);
        
        fprintf(fp, "rayleigh_data.xgrid\n");
        for (int j=0; j<MXRAYFF; j++) {
            int idx = i*MXRAYFF + j;
            fprintf(fp, "xgrid[%d][%d] = %10.5f\n", j, i,
                    rayleigh_data.xgrid[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "rayleigh_data.fcum\n");
        for (int j=0; j<MXRAYFF; j++) {
            int idx = i*MXRAYFF + j;
            fprintf(fp, "fcum[%d][%d] = %10.5f\n", j, i,
                    rayleigh_data.fcum[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "rayleigh_data.b_array\n");
        for (int j=0; j<MXRAYFF; j++) { // print just 5 first values
            int idx = i*MXRAYFF + j;
            fprintf(fp, "b_array[%d][%d] = %10.5f\n", j, i,
                    rayleigh_data.b_array[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "rayleigh_data.c_array\n");
        for (int j=0; j<MXRAYFF; j++) {
            int idx = i*MXRAYFF + j;
            fprintf(fp, "c_array[%d][%d] = %10.5f\n", j, i,
                    rayleigh_data.c_array[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "rayleigh_data.pmax\n");
        for (int j=0; j<MXGE; j++) {
            int idx = i*MXGE + j;
            fprintf(fp, "pmax0[%d][%d] = %10.5f, pmax1[%d][%d] = %10.5f\n",
                    j, i, rayleigh_data.pmax0[idx],
                    j, i, rayleigh_data.pmax1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "rayleigh_data.i_array\n");
        for (int j=0; j<RAYCDFSIZE; j++) {
            int idx = i*RAYCDFSIZE + j;
            fprintf(fp, "i_array[%d][%d] = %d\n", j, i,
                    rayleigh_data.i_array[idx]);
        }
        fprintf(fp, "\n");
        
    }
    fclose(fp);
}

/* Pair production definitions */

double fcoulc(double zi) {
    /* Calculates correction to Coulomb factor used in init_pair_data */
    
    double fc;
    double a = FSC*zi;
    
    /* a=fsc*Z,fcoulc=a^2((1+a^2)^-1+0.20206-0.0369*a^2+0.0083a^4-0.002a^6) */
    fc = 1 + pow(a, 2);
    fc = 1.0/fc;
    fc = fc + 0.20206 - 0.0369*pow(a, 2);
    fc = fc + 0.0083*pow(a, 4);
    fc = fc - 0.002*pow(a, 6);
    fc = fc*pow(a, 2);
    
    return fc;
}

double xsif(double zi, double fc) {
    
    /* Used in calculation of parameter of rejection function */
    double xi;
    
    if (zi == 4) {
        xi = 5.924/(4.710 - fc);
    }
    else if (zi == 3) {
        xi = 5.805/(4.740 - fc);
    }
    else if (zi == 2) {
        xi = 5.621/(4.790 - fc);
    }
    else if (zi == 1) {
        xi = 6.144/(5.310 - fc);
    }
    else {
        xi = log(1194.0*pow(zi,-2.0/3.0))/log(184.15*pow(zi,-1.0/3.0)) - fc;
    }
    
    return xi;
}

void initPairData() {
    /* The data calculated here corresponds partially to the subroutine
     fix_brems in egsnrc.macros. This subroutine calculates the parameters
     for the rejection function used in bremsstrahlung sampling*/

    double Zf, Zb, Zt, Zg, Zv; // medium functions, used for delx and delcm
    double fmax1, fmax2;
    
    /* Memory allocation */
    pair_data.dl1 = malloc(media.nmed*8*sizeof(double));
    pair_data.dl2 = malloc(media.nmed*8*sizeof(double));
    pair_data.dl3 = malloc(media.nmed*8*sizeof(double));
    pair_data.dl4 = malloc(media.nmed*8*sizeof(double));
    pair_data.dl5 = malloc(media.nmed*8*sizeof(double));
    pair_data.dl6 = malloc(media.nmed*8*sizeof(double));
    
    pair_data.bpar0 = malloc(media.nmed*sizeof(double));
    pair_data.bpar1 = malloc(media.nmed*sizeof(double));
    pair_data.delcm = malloc(media.nmed*sizeof(double));
    pair_data.zbrang = malloc(media.nmed*sizeof(double));
    
    int nmed = media.nmed;
    
    for (int imed=0; imed<nmed; imed++) {
        Zt = 0.0; Zb = 0.0; Zf = 0.0;
        
        for (int i=0; i<pegs_data.ne[imed]; i++) {
            
            /* Z of the corresponding element */
            double zi = pegs_data.elements[imed][i].z;
            
            /* Percentage of Z in a medium */
            double pi = pegs_data.elements[imed][i].pz;
            
            double fc = fcoulc(zi);  // Coulomb correction function
            double xi = xsif(zi, fc); // W.I.P Atomic electrons correction
            double aux = pi*zi*(zi + xi);
            Zt = Zt + aux;
            Zb = Zb - aux*log(zi)/3.0;
            Zf = Zf + aux*fc;
           
        }

        Zv = (Zb - Zf)/Zt;
        Zg = Zb/Zt;
        fmax1 = 2.0*(20.863 + 4.0*Zg) - 2.0*(20.029 + 4.0*Zg)/3.0;
        fmax2 = 2.0*(20.863 + 4.0*Zv) - 2.0*(20.029 + 4.0*Zv)/3.0;
        
        // The following data is used in brems.
        pair_data.dl1[imed*8 + 0] = (20.863 + 4.0*Zg)/fmax1;
        pair_data.dl2[imed*8 + 0] = -3.242/fmax1;
        pair_data.dl3[imed*8 + 0] = 0.625/fmax1;
        pair_data.dl4[imed*8 + 0] = (21.12 + 4.0*Zg)/fmax1;
        pair_data.dl5[imed*8 + 0] = -4.184/fmax1;
        pair_data.dl6[imed*8 + 0] = 0.952;
        
        pair_data.dl1[imed*8 + 1] = (20.029 + 4.0*Zg)/fmax1;
        pair_data.dl2[imed*8 + 1] = -1.93/fmax1;
        pair_data.dl3[imed*8 + 1] = -0.086/fmax1;
        pair_data.dl4[imed*8 + 1] = (21.12 + 4.0*Zg)/fmax1;
        pair_data.dl5[imed*8 + 1] = -4.184/fmax1;
        pair_data.dl6[imed*8 + 1] = 0.952;
        
        pair_data.dl1[imed*8 + 2] = (20.863 + 4.0*Zv)/fmax2;
        pair_data.dl2[imed*8 + 2] = -3.242/fmax2;
        pair_data.dl3[imed*8 + 2] = 0.625/fmax2;
        pair_data.dl4[imed*8 + 2] = (21.12 + 4.0*Zv)/fmax2;
        pair_data.dl5[imed*8 + 2] = -4.184/fmax2;
        pair_data.dl6[imed*8 + 2] = 0.952;
        
        pair_data.dl1[imed*8 + 3] = (20.029 + 4.0*Zv)/fmax2;
        pair_data.dl2[imed*8 + 3] = -1.93/fmax2;
        pair_data.dl3[imed*8 + 3] = -0.086/fmax2;
        pair_data.dl4[imed*8 + 3] = (21.12 + 4.0*Zv)/fmax2;
        pair_data.dl5[imed*8 + 3] = -4.184/fmax2;
        pair_data.dl6[imed*8 + 3] = 0.952;
        
        // The following data are used in pair production.
        pair_data.dl1[imed*8 + 4] = (3.0*(20.863 + 4.0*Zg) - (20.029 + 4.0*Zg));
        pair_data.dl2[imed*8 + 4] = (3.0*(-3.242) - (-1.930));
        pair_data.dl3[imed*8 + 4] = (3.0*(0.625) - (-0.086));
        pair_data.dl4[imed*8 + 4] = (2.0*21.12 + 8.0*Zg);
        pair_data.dl5[imed*8 + 4] = (2.0*(-4.184));
        pair_data.dl6[imed*8 + 4] = 0.952;
        
        pair_data.dl1[imed*8 + 5] = (3.0*(20.863 + 4.0*Zg) + (20.029 + 4.0*Zg));
        pair_data.dl2[imed*8 + 5] = (3.0*(-3.242) + (-1.930));
        pair_data.dl3[imed*8 + 5] = (3.0*0.625 + (-0.086));
        pair_data.dl4[imed*8 + 5] = (4.0*21.12 + 16.0*Zg);
        pair_data.dl5[imed*8 + 5] = (4.0*(-4.184));
        pair_data.dl6[imed*8 + 5] = 0.952;
        
        pair_data.dl1[imed*8 + 6] = (3.0*(20.863 + 4.0*Zv) - (20.029 + 4.0*Zv));
        pair_data.dl2[imed*8 + 6] = (3.0*(-3.242) - (-1.930));
        pair_data.dl3[imed*8 + 6] = (3.0*(0.625) - (-0.086));
        pair_data.dl4[imed*8 + 6] = (2.0*21.12 + 8.0*Zv);
        pair_data.dl5[imed*8 + 6] = (2.0*(-4.184));
        pair_data.dl6[imed*8 + 6] = 0.952;
        
        pair_data.dl1[imed*8 + 7] = (3.0*(20.863 + 4.0*Zv) + (20.029 + 4.0*Zv));
        pair_data.dl2[imed*8 + 7] = (3.0*(-3.242) + (-1.930));
        pair_data.dl3[imed*8 + 7] = (3.0*0.625 + (-0.086));
        pair_data.dl4[imed*8 + 7] = (4.0*21.12 + 16.0*Zv);
        pair_data.dl5[imed*8 + 7] = (4.0*(-4.184));
        pair_data.dl6[imed*8 + 7] = 0.952;
        
        pair_data.bpar1[imed] = pair_data.dl1[imed*8 + 6]/
            (3.0*pair_data.dl1[imed*8 + 7] + pair_data.dl1[imed*8 + 6]);
        pair_data.bpar0[imed] = 12.0 * pair_data.dl1[imed*8 +7]/
            (3.0*pair_data.dl1[imed*8 + 7] + pair_data.dl1[imed*8 + 6]);
        
        // The following is the calculation of the composite factor for angular
        // distributions, as carried out in $INITIALIZE-PAIR-ANGLE macro. It
        // corresponds to ( (1/111)*Zeff**(1/3) )**2
        double zbrang = 0.0;
        double pznorm = 0.0;
        
        for (int i = 0; i<pegs_data.ne[imed]; i++) {
            zbrang += (double)
            (pegs_data.elements[imed][i].pz)*
            (pegs_data.elements[imed][i].z)*
            ((pegs_data.elements[imed][i].z) + 1.0f);
            pznorm += pegs_data.elements[imed][i].pz;
        }
        pair_data.zbrang[imed] = (8.116224E-05)*pow(zbrang/pznorm, 1.0/3.0);
        pair_data.delcm[imed] = pegs_data.delcm[imed];
    }
    
    return;
}

void cleanPair() {
    
    free(pair_data.dl1);
    free(pair_data.dl2);
    free(pair_data.dl3);
    free(pair_data.dl4);
    free(pair_data.dl5);
    free(pair_data.dl6);
    
    free(pair_data.bpar0);
    free(pair_data.bpar1);
    free(pair_data.delcm);
    free(pair_data.zbrang);
    
    return;
}

void listPair() {
    
    /* Get file path from input data */
    char output_folder[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "output folder") != 1) {
        printf("Can not find 'output folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(output_folder, buffer);
    
    char file_name[256];
    strcpy(file_name, output_folder);
    strcat(file_name, "pair_data.lst");
    
    /* List pair data to output file */
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
    fprintf(fp, "Listing pair data: \n");
    for (int i=0; i<media.nmed; i++) {
        fprintf(fp, "For medium %s: \n", media.med_names[i]);
        
        fprintf(fp, "pair_data.delcm[%d] = %f\n", i, pair_data.delcm[i]);
        fprintf(fp, "pair_data.bpar0[%d] = %f\n", i, pair_data.bpar0[i]);
        fprintf(fp, "pair_data.bpar1[%d] = %f\n", i, pair_data.bpar1[i]);
        fprintf(fp, "pair_data.zbrang[%d] = %f\n", i, pair_data.zbrang[i]);
        
        fprintf(fp, "pair_data.dl1\n");
        for (int j=0; j<8; j++) {
            int idx = i*8 + j;
            fprintf(fp, "dl1[%d][%d] = %f\n", j, i, pair_data.dl1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "pair_data.dl2\n");
        for (int j=0; j<8; j++) {
            int idx = i*8 + j;
            fprintf(fp, "dl2[%d][%d] = %f\n", j, i, pair_data.dl2[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "pair_data.dl3\n");
        for (int j=0; j<8; j++) {
            int idx = i*8 + j;
            fprintf(fp, "dl3[%d][%d] = %f\n", j, i, pair_data.dl3[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "pair_data.dl4\n");
        for (int j=0; j<8; j++) {
            int idx = i*8 + j;
            fprintf(fp, "dl4[%d][%d] = %f\n", j, i, pair_data.dl4[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "pair_data.dl5 = \n");
        for (int j=0; j<8; j++) {
            int idx = i*8 + j;
            fprintf(fp, "dl5[%d][%d] = %f\n", j, i, pair_data.dl5[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "pair_data.dl6 = \n");
        for (int j=0; j<8; j++) {
            int idx = i*8 + j;
            fprintf(fp, "dl6[%d][%d] = %f\n", j, i, pair_data.dl6[idx]);
        }
        fprintf(fp, "\n");
        
    }
    
    fclose(fp);
    
    return;
}

/*******************************************************************************
/* Electron physical processes definitions
*******************************************************************************/
struct Electron electron_data;

void shower() {
 
    while (stack.np >= 0) {
        if (stack.iq[stack.np] == 0) {
            photon();
        } else {
            electron();
        }
    }
    
    return;
}

/* Media definitions */
struct Media media;

struct Pegs pegs_data;

void initMediaData(){
    
    /* Array that indicates if medium was found on pegs file or not */
    int *media_found = (int*) malloc(media.nmed*sizeof(int));
    
    /* Get media information from pegs file */
    int nmedia_found;
    nmedia_found = readPegsFile(media_found);
    
    /* Check if all requested media was found */
    if (nmedia_found < media.nmed) {
        printf("The following media were not found or could not be read "
               "from pegs file:\n");
        for (int i=0; i<media.nmed; i++) {
            if (media_found[i] == 0) {
                printf("\t %s\n", media.med_names[i]);
            }
        }
        exit(EXIT_FAILURE);
    }
    
    /* Initialize the photon data using the specified cross-section files */
    initPhotonData();
    
    /* Initialize data needed for Rayleigh and Pair production interactions */
    initRayleighData();
    initPairData();
    
    /* Initialize data needed for electron multi-scattering interactions */
    initMscatData();
    
    printf("Interaction data initialized successfully\n");
    
    /* Cleaning */
    free(media_found);
    
    return;
}

int readPegsFile(int *media_found) {
    
    /* Get file path from input data */
    char pegs_file[BUFFER_SIZE];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "pegs file") != 1) {
        printf("Can not find 'pegs file' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(pegs_file, buffer);
    
    /* Open pegs file */
    FILE *fp;
    
    if ((fp = fopen(pegs_file, "r")) == NULL) {
        printf("Unable to open file: %s\n", pegs_file);
        exit(EXIT_FAILURE);
    }

    printf("Path to pegs file : %s\n", pegs_file);

    int nmedia = 0; // number of media found on pegs4 file
    
    /* The following data is used in electron transport modeling.
     Allocate Electron struct arrays */
    electron_data.blcc = malloc(media.nmed*sizeof(double));
    electron_data.xcc = malloc(media.nmed*sizeof(double));
    electron_data.eke0 = malloc(media.nmed*sizeof(double));
    electron_data.eke1 = malloc(media.nmed*sizeof(double));
    electron_data.esig0 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.esig1 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.psig0 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.psig1 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.ededx0 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.ededx1 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.pdedx0 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.pdedx1 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.ebr10 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.ebr11 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.pbr10 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.pbr11 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.pbr20 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.pbr21 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.tmxs0 = malloc(media.nmed*MXEKE*sizeof(double));
    electron_data.tmxs1 = malloc(media.nmed*MXEKE*sizeof(double));
    
    /* Zero the following arrays, as they are surely not totally used. */
    memset(electron_data.esig0, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.esig1, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.psig0, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.psig1, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.ededx0, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.ededx1, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.pdedx0, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.pdedx1, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.ebr10, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.ebr11, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.pbr10, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.pbr11, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.pbr20, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.pbr21, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.tmxs0, 0.0, media.nmed*MXEKE*sizeof(double));
    memset(electron_data.tmxs1, 0.0, media.nmed*MXEKE*sizeof(double));
    
    do {
        /* Read a line of pegs file */
        char buffer[BUFFER_SIZE];
        fgets(buffer, BUFFER_SIZE, fp);
        
        /* Here starts a medium definition */
        if (strstr(buffer, " MEDIUM=") == buffer) {
            char name_with_spaces[25];
            int c = 0;
            while (c < 24) {
                name_with_spaces[c] = buffer[c + 8];
                c++;
            }
            
            /* Next algorithm take out spaces */
            name_with_spaces[c] = '\0';
            char name[25];
            int j = 0;
            /* Read name up to first space */
            for (int i = 0; i < 24; i++) {
                if (name_with_spaces[i] != ' ') {
                    name[j] = name_with_spaces[i];
                    j++;
                }
                else
                    break;
            }
            name[j] = '\0';
            
            /* see whether this is required medium comparing with the medium
             list */
            int required = 0;
            int imed = 0; // medium index
            
            for (int i = 0; i < media.nmed; i++) {
                char cname[20];
                strncpy(cname, media.med_names[i], 20);
                if (strcmp(name, cname) == 0) {
                    required = 1;
                    imed = i;
                    break;
                }
            }
            if (required == 0) { // return to beginning of the do loop
                continue;
            }
            
            /* We have found the i'th required medium */
            strncpy(pegs_data.names[imed], name, 25);
            pegs_data.ne[imed] = 0;
            
            /* Read the next line containing the density, number of elements
             and flags */
            fgets(buffer, BUFFER_SIZE, fp);
            int ok = 1;
            char s[BUFFER_SIZE];
            char s2[BUFFER_SIZE];
            char* temp;
            strcpy(s, buffer);
            char* token = strtok_r(s, ",", &temp);
            int i = 0;
            char* name2[20];
            char* name3 = NULL;
            char* value;
            
            while (token) {
                char *temp2 = token;
                name2[i] = temp2;
                strcpy(s2, name2[i]);
                char* temp4;
                char* token2 = strtok_r(s2, "=", &temp4);
                for (int k = 0; k<2; k++) {
                    char *temp3 = token2;
                    if (k == 0) {
                        name3 = temp3;
                    }
                    else {
                        value = temp3;
                    }
                    token2 = strtok_r(NULL, "=", &temp4);
                }
                
                /* Next algorithm take out spaces */
                char* tempname = name3;
                int l = 0;
                for (int i = 0; 1; i++) {
                    /* The loop should work as an infinite loop that breaks
                     when the word ends */
                    if (tempname[i] != ' ' && tempname[i] != '\0') {
                        name3[l] = tempname[i];
                        l++;
                    }
                    else if (l>1) {
                        name3[l] = '\0';
                        break;
                    }
                    else if (strlen(tempname) + 1 == i) {
                        printf(" Error reading %s", tempname);
                        break;
                    }
                }
                
                if (strcmp(name3, "RHO") == 0) {
                    double d;
                    if (sscanf(value, "%lf", &d) != 1) {
                        ok = 0;
                        break;
                    }
                    pegs_data.rho[imed] = d;
                }
                else if (strcmp(name3, "NE") == 0) {
                    int u;
                    if (sscanf(value, "%u", &u) != 1) {
                        ok = 0;
                        break;
                    }
                    pegs_data.ne[imed] = u;
                }
                else if (strcmp(name3, "IUNRST") == 0) {
                    int i;
                    if (sscanf(value, "%d", &i) != 1) {
                        ok = 0;
                        break;
                    }
                    pegs_data.iunrst[imed] = i;
                }
                else if (strcmp(name3, "EPSTFL") == 0) {
                    int i;
                    if (sscanf(value, "%d", &i) != 1) {
                        ok = 0;
                        break;
                    }
                    pegs_data.epstfl[imed] = i;
                }
                else if (strcmp(name3, "IAPRIM") == 0) {
                    int i;
                    if (sscanf(value, "%d", &i) != 1) {
                        ok = 0;
                        break;
                    }
                    pegs_data.iaprim[imed] = i;
                }
                token = strtok_r(NULL, ",", &temp);
                i++;
            }
            if (!ok) {
                continue;
            } // end of 2nd line readings
            
            /* Read elements, same algorithm */
            for (int m = 0; m < pegs_data.ne[imed]; m++) {
                struct Element element = { 0 };
                fgets(buffer, BUFFER_SIZE, fp);
                char s[BUFFER_SIZE];
                char s2[BUFFER_SIZE];
                char* temp;
                strcpy(s, buffer);
                char* token = strtok_r(s, ",", &temp);
                int i = 0;
                int ok = 1;
                char* name2[20];    // array of strings, stores reading of
                // "NAME = VALUE" format
                char* name3 = NULL; // array of strings, stores name of
                // property
                char* value = NULL; // array of strings, stores the value
                // corresponding to the name
                
                while (token) {
                    char *temp2 = token;
                    name2[i] = temp2;
                    strcpy(s2, name2[i]);
                    char* temp4;
                    char* token2 = strtok_r(s2, "=", &temp4);
                    for (int k = 0; k<2; k++) {
                        char *temp3 = token2;
                        if (k == 0) {
                            name3 = temp3;
                        }
                        else {
                            value = temp3;
                        }
                        token2 = strtok_r(NULL, "=", &temp4);
                    }
                    char* tempname = name3;
                    int l = 0;
                    for (int i = 0; i < (int)strlen(tempname) + 2; i++) {
                        if (tempname[i] != ' ' && tempname[i] != '\0') {
                            name3[l] = tempname[i];
                            l++;
                        }
                        else if (tempname[i] == '\0') {
                            name3[l] = tempname[i];
                            break;
                        }
                        else if (l>1) {
                            name3[l] = '\0';
                            break;
                        }
                    }
                    
                    if (strcmp(name3, "ASYM") == 0) {
                        if (strlen(value) < 2) {
                            ok = 0;
                            break;
                        }
                        element.symbol[0] = value[0];
                        element.symbol[1] = value[1];
                        element.symbol[2] = '\0';
                    }
                    else if (strcmp(name3, "Z") == 0) {
                        double d;
                        if (sscanf(value, "%lf", &d) != 1) {
                            ok = 0;
                            break;
                        }
                        element.z = d;
                    }
                    else if (strcmp(name3, "A") == 0) {
                        double d;
                        if (sscanf(value, "%lf", &d) != 1) {
                            ok = 0;
                            break;
                        }
                        element.wa = d;
                    }
                    else if (strcmp(name3, "PZ") == 0) {
                        double d;
                        if (sscanf(value, "%lf", &d) != 1) {
                            ok = 0;
                            break;
                        }
                        element.pz = d;
                    }
                    else if (strcmp(name3, "RHOZ") == 0) {
                        double d;
                        if (sscanf(value, "%lf", &d) != 1) {
                            ok = 0;
                            break;
                        }
                        element.rhoz = d;
                        
                    }
                    token = strtok_r(NULL, ",", &temp);
                    i++;
                    
                }
                if (ok == 0) {
                    break;
                }
                
                pegs_data.elements[imed][m] = element;
            }
            
            if (ok == 0) {
                continue;
            }
            
            /* Read next line that contines rlc, ae, ap, ue, up */
            fgets(buffer, BUFFER_SIZE, fp);
            
            /* The format specifier '%lf' is needed to correctly recognize
             engineering notation. I do not now if this is a property of
             clang, because I had not to do that before */
            if (sscanf(buffer, "%lf %lf %lf %lf %lf\n", &pegs_data.rlc[imed],
                       &pegs_data.ae[imed], &pegs_data.ap[imed],
                       &pegs_data.ue[imed], &pegs_data.up[imed]) != 5) {
                continue;
            }
            pegs_data.te[imed] = pegs_data.ae[imed] - RM;
            pegs_data.thmoll[imed] = (pegs_data.te[imed]) * 2 + RM;
            
            /* Save the medium and mark it found */
            fgets(buffer, BUFFER_SIZE, fp);
            if (sscanf(buffer, "%d %d %d %d %d %d %d\n",
                       &pegs_data.msge[imed], &pegs_data.mge[imed],
                       &pegs_data.mseke[imed], &pegs_data.meke[imed],
                       &pegs_data.mleke[imed], &pegs_data.mcmfp[imed],
                       &pegs_data.mrange[imed]) != 7) {
                continue;
            }
            if (pegs_data.meke[imed]>MXEKE) {
                printf("Medium %d has MEKE too big, change MXEKE to %d in "
                       "source code", imed, pegs_data.meke[imed]);
                continue;
            }
            
            for (int i = 0; i<7; i++) {
                fgets(buffer, BUFFER_SIZE, fp);
            }
            double del1, del2, del3, del4, del5;
            if (sscanf(buffer, "%lf %lf %lf %lf %lf ",
                       &del1, &del2, &del3, &del4, &del5) != 5) {
                continue;
            }
            
            /* The parameter delcm will be transferred in the initialization
             of pair production */
            double dl6, ALPHI1, BPAR1, DELPOS1, ALPHI2, BPAR2, DELPOS2, XR0, TEFF0;
            fscanf(fp, "%lf ", &dl6);
            fscanf(fp, "%lf %lf %lf %lf %lf ", &pegs_data.delcm[imed], &ALPHI1,
                   &ALPHI2, &BPAR1, &BPAR2);
            fscanf(fp, "%lf %lf ", &DELPOS1, &DELPOS2);
            fscanf(fp, "%lf %lf %lf %lf ", &XR0, &TEFF0, &electron_data.blcc[imed],
                   &electron_data.xcc[imed]);
            fscanf(fp, "%lf %lf ", &electron_data.eke0[imed],
                   &electron_data.eke1[imed]);
            
            int neke = pegs_data.meke[imed];
            for (int k = 0; k<neke; k++) {
                fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf ",
                       &electron_data.esig0[imed*MXEKE + k],
                       &electron_data.esig1[imed*MXEKE + k],
                       &electron_data.psig0[imed*MXEKE + k],
                       &electron_data.psig1[imed*MXEKE + k],
                       &electron_data.ededx0[imed*MXEKE + k],
                       &electron_data.ededx1[imed*MXEKE + k],
                       &electron_data.pdedx0[imed*MXEKE + k],
                       &electron_data.pdedx1[imed*MXEKE + k]);
                
                fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf ",
                       &electron_data.ebr10[imed*MXEKE + k],
                       &electron_data.ebr11[imed*MXEKE + k],
                       &electron_data.pbr10[imed*MXEKE + k],
                       &electron_data.pbr11[imed*MXEKE + k],
                       &electron_data.pbr20[imed*MXEKE + k],
                       &electron_data.pbr21[imed*MXEKE + k],
                       &electron_data.tmxs0[imed*MXEKE + k],
                       &electron_data.tmxs1[imed*MXEKE + k]);
            }
            
            /* length units, only for cm */
            double DFACTI = 1.0 / (pegs_data.rlc[imed]);
            electron_data.blcc[imed] *= DFACTI;
            for (int k = 0; k<neke; k++) {
                electron_data.esig0[imed*MXEKE + k] *= DFACTI;
                electron_data.psig0[imed*MXEKE + k] *= DFACTI;
                electron_data.ededx0[imed*MXEKE + k] *= DFACTI;
                electron_data.pdedx0[imed*MXEKE + k] *= DFACTI;
                electron_data.pdedx1[imed*MXEKE + k] *= DFACTI;
                electron_data.esig1[imed*MXEKE + k] *= DFACTI;
                electron_data.psig1[imed*MXEKE + k] *= DFACTI;
                electron_data.ededx1[imed*MXEKE + k] *= DFACTI;
            }
            electron_data.xcc[imed] *= sqrt(DFACTI);
            
            /* Mark the medium found */
            media_found[imed] = 1;
            nmedia++;
        }
    } while ((nmedia < media.nmed) && !feof(fp));
    
    /* Print some information for debugging purposes */
    if(verbose_flag) {
        for (int i=0; i<media.nmed; i++) {
            printf("For medium %s: \n", pegs_data.names[i]);
            printf("\t ne = %d\n", pegs_data.ne[i]);
            printf("\t rho = %f\n", pegs_data.rho[i]);
            printf("\t iunrst = %d\n", pegs_data.iunrst[i]);
            printf("\t epstfl = %d\n", pegs_data.epstfl[i]);
            printf("\t iaprim = %d\n", pegs_data.iaprim[i]);
            
            printf("\t ae = %f\n", pegs_data.ae[i]);
            printf("\t ap = %f\n", pegs_data.ap[i]);
            
            printf("\t msge = %d\n", pegs_data.msge[i]);
            printf("\t mge = %d\n", pegs_data.mge[i]);
            printf("\t mseke = %d\n", pegs_data.mseke[i]);
            printf("\t meke = %d\n", pegs_data.meke[i]);
            printf("\t mleke = %d\n", pegs_data.mleke[i]);
            printf("\t mcmfp = %d\n", pegs_data.mcmfp[i]);
            printf("\t mrange = %d\n", pegs_data.mrange[i]);
            printf("\t delcm = %f\n", pegs_data.delcm[i]);
        
            for (int j=0; j<pegs_data.ne[i]; j++) {
                printf("\t For element %s inside %s: \n",
                       pegs_data.elements[i][j].symbol, pegs_data.names[i]);
                printf("\t\t z = %f\n", pegs_data.elements[i][j].z);
                printf("\t\t wa = %f\n", pegs_data.elements[i][j].wa);
                printf("\t\t pz = %f\n", pegs_data.elements[i][j].pz);
                printf("\t\t rhoz = %f\n", pegs_data.elements[i][j].rhoz);
            }
        }
    
    }
    
    /* Close pegs file */
    fclose(fp);
    
    return nmedia;
}

/******************************************************************************/
