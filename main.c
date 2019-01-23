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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/******************************************************************************/
/* Parsing program options with getopt long
 http://www.gnu.org/software/libc/manual/html_node/Getopt.html#Getopt */
#ifndef MATLAB_COMPILE_MEX
#include <getopt.h>
#endif

/* Flag set by ‘--verbose’. */
static int verbose_flag;

/******************************************************************************/

/******************************************************************************/
/* A simple C/C++ class to parse input files and return requested
 key value -- https://github.com/bmaynard/iniReader */

#include <string.h>
#include <ctype.h>

/* Parse a configuration file */
void parseInputFile(char *file_name);

/* Copy the value of the selected input item to the char pointer */
int getInputValue(char *dest, char *key);

/* Returns nonzero if line is a string containing only whitespace or is empty */
int lineBlack(char *line);

/* Remove white spaces from string str_untrimmed and saves the results in
 str_trimmed. Useful for string input values, such as file names */
void removeSpaces(char* str_trimmed,
                  const char* str_untrimmed);

struct inputItems {
    char key[255];    
    char value[255];
};

struct inputItems input_items[80];  // i.e. support 80 key,value pairs
int input_idx = 0;                // number of key,value pairs

/******************************************************************************/

/******************************************************************************/
/* Geometry definition */
#define MXMED 9 // maximum number of media supported

struct Geom {
    int nmed;                   // number of media in phantom file
    char med_names[MXMED][60];  // media names (as found in .pegs4dat file)
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

void initPhantom(void);
void cleanPhantom(void);

/******************************************************************************/
/* Source definition */
#define MXEBIN 200      // number of energy bins of spectrum
#define INVDIM 1000     // number of bins in inverse CDF

struct Source {
    int nmed;                   // number of media in phantom file
    int spectrum;               // 0 : monoenergetic, 1 : spectrum
    int charge;                 // 0 : photons, -1 : electron, +1 : positron
    
    /* For monoenergetic source */
    double energy;
    
    /* For spectrum */
    double deltak;              // number of elements in inverse CDF
    double *cdfinv1;            // energy value of bin
    double *cdfinv2;            // prob. that particle has energy xi
    
    /* Beamlets shape information */
    int nbeamlets;               // number of beamlets per beam
    int *ibeam;                  // index of beam per beamlet
    
    double *xsource;           // coordinates of the source of each beam
    double *ysource;          
    double *zsource;          
        
    double *xcorner;           // coordinates of the bixel corner
    double *ycorner;           
    double *zcorner;  
    
    double *xside1;           // coordinates of the first side of bixel
    double *yside1;           
    double *zside1;
    
    double *xside2;           // coordinates of the second side of bixel
    double *yside2;           
    double *zside2;
        
};
struct Source source;

void initSource(void);
void cleanSource(void);

/******************************************************************************/
/* Score definition */
struct Score {
    double ensrc;               // total energy from source
    double *endep;              // 3D dep. energy matrix per batch
    
    /* The following variables are needed for statistical analysis. Their
     values are accumulated across the simulation */
    double *accum_endep;        // 3D deposited energy matrix
    double *accum_endep2;       // 3D square deposited energy
};
struct Score score;

void initScore(void);
void cleanScore(void);
void ausgab(double edep);
void accumEndep(void);

void accumulateResults(int iout, int nhist, int nbatch);
void outputResults(char *output_file, int iout, int nhist, int nbatch);

/******************************************************************************/
/* Stack definition */
#define MXSTACK 100  // maximum number of particles on stack

struct Stack {
    int np;         // stack pointer
    
    int *iq;        // particle charge
    int *ir;        // current region
    
    double *e;      // total particle energy
    
    double *x;      // particle coordinates
    double *y;
    double *z;
    
    double *u;      // particle direction cosines
    double *v;
    double *w;
    
    double *dnear;  // perpendicular distance to nearest boundary
    double *wt;     // particle weight
};
struct Stack stack;

void initStack(void);
void initHistory(int ibeamlet);
void cleanStack(void);

/******************************************************************************/
/* Region-by-region data definition */
#define VACUUM -1

struct Region {
    int *med;
    double *rhof;
    double *pcut;
    double *ecut;
};
struct Region region;

void initRegions(void);
void cleanRegions(void);

/******************************************************************************/
/* Random number generator definition */
#define NRANDOM 128

struct Random {
    int crndm;
    int cdrndm;
    int cmrndm;
    int ixx;
    int jxx;
    int rng_seed;
    int *urndm;
    int *rng_array;
    double twom24;
};
struct Random rng;

void initRandom(void);
void getRandom(void);
double setRandom(void);
void cleanRandom(void);

/******************************************************************************/
/* Electromagnetic shower simulation */

struct Uphi {
    /* This structure holds data saved between uphi() calls */
    double A, B, C;
    double cosphi, sinphi;
};

struct Mscats {
    /* This structure holds data saved between mscat calls */
    int i;
    int j;
    double omega2;
};

struct Spinr {
    /* This structure holds data saved between spinRejection calls */
    int i;
    int j;
};

void shower(void);
void transferProperties(int npnew, int npold);
void uphi21(struct Uphi *uphi, double costhe, double sinthe);
void uphi32(struct Uphi *uphi, double costhe, double sinthe);

/******************************************************************************/
/* Media definition */
#define RM 0.5109989461 // MeV * c^(-2)
#define MXELEMENT 50    // maximum number of elements in a medium
#define MXEKE 500

struct Element {
    /* Attributes of an element in a medium */
    char symbol[3];
    double z;
    double wa;
    double pz;
    double rhoz;
    
};

struct Pegs {
    /* Data extracted from pegs file */
    char names[MXMED][60];          // media names (as found in .pegs4dat file)
    int ne[MXMED];                  // number of elements in medium
    int iunrst[MXMED];              // flag for type of stopping power
    int epstfl[MXMED];              // flag for ICRU37 collision stopping powers
    int iaprim[MXMED];              // flag for ICRU37 radiative stopping powers
    
    int msge[MXMED];
    int mge[MXMED];
    int mseke[MXMED];
    int meke[MXMED];
    int mleke[MXMED];
    int mcmfp[MXMED];
    int mrange[MXMED];
    
    double rho[MXMED];              // mass density of medium
    double rlc[MXMED];              // radiation length for the medium (in cm)
    double ae[MXMED], ap[MXMED];    // electron and photon creation threshold E
    double ue[MXMED], up[MXMED];    // upper electron and photon energy
    double te[MXMED];
    double thmoll[MXMED];
    double delcm[MXMED];
    
    struct Element elements[MXMED][MXELEMENT];  // element properties
};
struct Pegs pegs_data;

void initMediaData(void);
int readPegsFile(int *media_found);

/******************************************************************************/
/* Photon data definition */
#define MXGE 2000       // gamma mapped energy intervals

struct Photon {
    double *ge0, *ge1;
    double *gmfp0, *gmfp1;
    double *gbr10, *gbr11;
    double *gbr20, *gbr21;
    double *cohe0, *cohe1;
};
struct Photon photon_data;

void initPhotonData(void);
void readXsecData(char *file, int *ndat,
                  double **xsec_data0,
                  double **xsec_data1);

void heap_sort(int n, double *values, int *indices);
double *get_data(int flag, int ne, int *ndat,
                 double **data0, double **data1,
                 double *z_sorted, double *pz_sorted,
                 double ge0, double ge1);
double kn_sigma0(double e);
void cleanPhoton(void);
void listPhoton(void);

/******************************************************************************/
/* Rayleigh data definition */
#define MXRAYFF 100     // Rayleigh atomic form factor
#define RAYCDFSIZE 100  // CDF from Rayleigh from factors squared
#define HC_INVERSE  80.65506856998
#define TWICE_HC2   0.000307444456

struct Rayleigh {
    double *xgrid;
    double *fcum;
    double *b_array;
    double *c_array;
    double *pe_array;
    double *pmax0;
    double *pmax1;
    int *i_array;
};
struct Rayleigh rayleigh_data;

void initRayleighData(void);
void readFfData(double *xval, double **aff);
void cleanRayleigh(void);
void listRayleigh(void);

/******************************************************************************/
/* Pair data definition */
#define FSC 0.00729735255664 // fine structure constant

struct Pair {
    double *dl1;
    double *dl2;
    double *dl3;
    double *dl4;
    double *dl5;
    double *dl6;
    
    double *bpar0;
    double *bpar1;
    double *delcm;
    double *zbrang;
};

struct Pair pair_data;

void initPairData(void);
double fcoulc(double zi);
double xsif(double zi, double fc);
void cleanPair(void);
void listPair(void);

/******************************************************************************/
/* Photon transport process */
#define SGMFP 1E-05 // smallest gamma mean free path

void photon(void);

void rayleigh(int imed, double eig, double gle, int lgle);

double setPairRejectionFunction(int imed, double xi, double esedei,
                                double eseder, double tteig);
void pair(int imed);

void compton(void);

void photo(void);

/******************************************************************************/
/* Electron data definition */
#define XIMAX 0.5
#define ESTEPE 0.25
#define EPSEMFP 1.0E-5  // smallest electron mean free path
#define SKIN_DEPTH_FOR_BCA 3


struct Electron {
    double *esig0;
    double *esig1;
    double *psig0;
    double *psig1;
    
    double *ededx0;
    double *ededx1;
    double *pdedx0;
    double *pdedx1;
    
    double *ebr10;
    double *ebr11;
    double *pbr10;
    double *pbr11;
    
    double *pbr20;
    double *pbr21;
    
    double *tmxs0;
    double *tmxs1;
    
    double *blcce0;
    double *blcce1;
    
    double *etae_ms0;
    double *etae_ms1;
    double *etap_ms0;
    double *etap_ms1;
    
    double *q1ce_ms0;
    double *q1ce_ms1;
    double *q1cp_ms0;
    double *q1cp_ms1;
    
    double *q2ce_ms0;
    double *q2ce_ms1;
    double *q2cp_ms0;
    double *q2cp_ms1;
    
    double *range_ep;
    double *e_array;
    
    double *eke0;
    double *eke1;
    
    int *sig_ismonotone;
    
    double *esig_e;
    double *psig_e;
    double *xcc;
    double *blcc;
    double *expeke1;
    
};
struct Electron electron_data;

/* Screened Rutherford MS data */
#define MXL_MS 63
#define MXQ_MS 7
#define MXU_MS 31
#define LAMBMIN_MS 1.0
#define LAMBMAX_MS 1.0E5
#define QMIN_MS 1.0E-3
#define QMAX_MS 0.5

struct Mscat {

    double *ums_array;
    double *fms_array;
    double *wms_array;
    int *ims_array;
    
    double dllambi;
    double dqmsi;
};

struct Mscat mscat_data;

/* Spin data */
#define MXE_SPIN 15
#define MXE_SPIN1 2*MXE_SPIN+1
#define MXQ_SPIN 15
#define MXU_SPIN 31

struct Spin {
    double b2spin_min;
    double dbeta2i;
    double espml;
    double dleneri;
    double dqq1i;
    double *spin_rej;
};

struct Spin spin_data;

void initMscatData(void);
void cleanElectron(void);
void listElectron(void);

void readRutherfordMscat(int nmed);
void cleanMscat(void);
void listMscat(void);

void initSpinData(int nmed);
void cleanSpin(void);
void listSpin(void);

void setSpline(double *x, double *f, double *a, double *b, double *c,
                double *d,int n);
double spline(double s, double *x, double *a, double *b, double *c,
              double *d, int n);

/******************************************************************************/
/* Electron interaction processes */
void electron(void);
double computeDrange(int imed, int iq, int lelke, double ekei, double ekef, 
    double elkei, double elkef);
double computeEloss(int imed, int iq, int irl, double rhof, 
    double tustep, double range, double eke, double elke, int lelke);
double msdist(int imed, int iq, double rhof, double de, double tustep, 
    double eke, double *x_final, double *y_final, double *z_final, 
    double *u_final, double *v_final, double *w_final);
void mscat(int imed, int qel, int *spin_index, int *find_index, 
    double elke, double beta2, double q1,  double lambda, double chia2, 
    double *cost, double *sint, struct Mscats *m_scat, struct Spinr *spin_r);
double spinRejection(int imed, int qel,	double elke, double beta2, double q1,
	double cost, int *spin_index, int is_single, struct Spinr *spin_r);
void sscat(int imed, int qel, double chia2, double elke, double beta2,
	double *cost, double *sint);
void rannih(void);
void brems(void);
void moller(void);
void bhabha(void);
void annih(void);

/******************************************************************************/
/* Auxiliary functions during simulation */
int pwlfInterval(int idx, double lvar, double *coef1, double *coef0);
double pwlfEval(int idx, double lvar, double *coef1, double *coef0);

/******************************************************************************/
/* Geometry functions */
void howfar(int *idisc, int *irnew, double *ustep);
double hownear(void);

/******************************************************************************/
/* ompMC main function */
#ifndef MATLAB_COMPILE_MEX
int main (int argc, char **argv) {
    
    /* Execution time measurement */
    clock_t tbegin, tend;
    tbegin = clock();
    
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
    
    /* Read geometry information from phantom file and initialize geometry */
    initPhantom();
    
    /* With number of media and media names initialize the medium data */
    initMediaData();
    
    /* Initialize radiation source */
    initSource();
    
    /* Initialize data on a region-by-region basis */
    initRegions();
    
    /* Preparation of scoring struct */
    initScore();
    
    /* Initialize random number generator */
    initRandom();
    
    /* Initialize particle stack */
    initStack();
    
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
    char buffer[128];
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
    printf("Execution time up to this point : %8.5f seconds\n",
           (double)(clock() - tbegin)/CLOCKS_PER_SEC);
    
    for (int ibatch=0; ibatch<nbatch; ibatch++) {
        if (ibatch == 0) {
            /* Print header for information during simulation */
            printf("%-10s\t%-15s\t%-10s\n", "Batch #", "Elapsed time",
                   "RNG state");
            printf("%-10d\t%-15.5f\t%-5d%-5d\n", ibatch,
                   (double)(clock() - tbegin)/CLOCKS_PER_SEC, rng.ixx, rng.jxx);
        }
        else {
            /* Print state of current batch */
            printf("%-10d\t%-15.5f\t%-5d%-5d\n", ibatch,
                   (double)(clock() - tbegin)/CLOCKS_PER_SEC, rng.ixx, rng.jxx);
            
        }
        
        for (int ihist=0; ihist<nperbatch; ihist++) {
            /* Initialize particle history */
            initHistory(ihist);
            
            /* Start electromagnetic shower simulation */
            shower();
        }
        
        /* Accumulate results of current batch for statistical analysis */
        accumEndep();
    }
    
    /* Print some output and execution time up to this point */
    printf("Simulation finished\n");
    printf("Execution time up to this point : %8.5f seconds\n",
           (double)(clock() - tbegin)/CLOCKS_PER_SEC);
    
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
    cleanRandom();
    cleanScore();
    cleanStack();
    
    /* Get total execution time */
    tend = clock();
    printf("Total execution time : %8.5f seconds\n",
           (double)(tend - tbegin)/CLOCKS_PER_SEC);
    
    exit (EXIT_SUCCESS);
}
#endif
void parseInputFile(char *input_file) {
    
    char buf[120];      // support lines up to 120 characters
    
    /* Make space for the new string */
    char *extension = ".inp";
    char* file_name = malloc(strlen(input_file) + strlen(extension) + 1);
    strcpy(file_name, input_file);
    strcat(file_name, extension); /* add the extension */
    
    FILE *fp;
    if ((fp = fopen(file_name, "r")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
    while (fgets(buf, sizeof(buf), fp) != NULL) {
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
    
    return;
}

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

void initPhantom() {
    
    /* Get phantom file path from input data */
    char phantom_file[128];
    char buffer[128];
    
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
    fgets(buffer, sizeof(buffer), fp);
    geometry.nmed = atoi(buffer);
    
    /* Get media names on phantom file */
    for (int i=0; i<geometry.nmed; i++) {
        fgets(buffer, sizeof(buffer), fp);
        removeSpaces(geometry.med_names[i], buffer);
    }
    
    /* Skip next line, it contains dummy input */
    fgets(buffer, sizeof(buffer), fp);
    
    /* Read voxel numbers on each direction */
    fgets(buffer, sizeof(buffer), fp);
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
    fgets(buffer, sizeof(buffer), fp);
    
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
            fgets(buffer, sizeof(buffer), fp);
        }
        /* Skip blank line */
        fgets(buffer, sizeof(buffer), fp);
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
        fgets(buffer, sizeof(buffer), fp);
    }
    
    /* Summary with geometry information */
    printf("Number of media in phantom : %d\n", geometry.nmed);
    printf("Media names: ");
    for (int i=0; i<geometry.nmed; i++) {
        printf("%s, ", geometry.med_names[i]);
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
    #ifndef MATLAB_COMPILE_MEX
    free(geometry.med_indices);
    free(geometry.med_densities);
    #endif
    return;
}

void initSource() {
    
    /* Get spectrum file path from input data */
    char spectrum_file[128];
    char buffer[128];
    
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
        fgets(buffer, sizeof(buffer), fp);
        printf("Spectrum file title: %s", buffer);
        
        /* Read number of bins and spectrum type */
        double enmin;   /* lower energy of first bin */
        int nensrc;     /* number of energy bins in spectrum histogram */
        int imode;      /* 0 : histogram counts/bin, 1 : counts/MeV*/
        
        fgets(buffer, sizeof(buffer), fp);
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
            fgets(buffer, sizeof(buffer), fp);
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
    }
    else {  /* monoenergetic source */
        if (getInputValue(buffer, "mono energy") != 1) {
            printf("Can not find 'mono energy' key on input file.\n");
            exit(EXIT_FAILURE);
        }
        source.energy = atof(buffer);
        printf("%f monoenergetic source\n", source.energy);
        
    }
    
    /* TODO: Initialize geometrical data of the source */
    
    
    return;
}

void cleanSource() {
    
    free(source.cdfinv1);
    free(source.cdfinv2);
    
    return;
}

void initRegions() {
    
    /* +1 : consider region surrounding phantom */
    int nreg = geometry.isize*geometry.jsize*geometry.ksize + 1;
    
    /* Allocate memory for region data */
    region.med = malloc(nreg*sizeof(int));
    region.rhof = malloc(nreg*sizeof(double));
    region.pcut = malloc(nreg*sizeof(double));
    region.ecut = malloc(nreg*sizeof(double));
    
    /* First get global energy cutoff parameters */
    char buffer[128];
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

    /* Print some information for debugging purposes */
    if (verbose_flag) {
        printf("Listing region data: \n");
        for (int i=0; i<5; i++) {
            printf("For region %d\n", i);
            printf("\t med = %d\n", region.med[i]);
            printf("\t rhof = %f\n", region.rhof[i]);
            printf("\t pcut = %f\n", region.pcut[i]);
            printf("\t ecut = %f\n", region.ecut[i]);
        }
    }
    
    return;
}

void cleanRegions() {
    
    free(region.med);
    free(region.rhof);
    free(region.ecut);
    free(region.pcut);
    
    return;
}

void initRandom() {
    
    /* Get initial seeds from input */
    char buffer[128];
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


/* The following set of uphi functions set coordinates for new particle or
 reset direction cosines of old one. Generate random azimuth selection and
 replace the direction cosine with their new values. */

void uphi21(struct Uphi *uphi,
            double costhe, double sinthe) {
    /* This section is used if costhe and sinthe are already known. Phi
     is selected uniformly over the interval (0,2Pi) */
    int np = stack.np;
    double r1 = setRandom();
    double phi = 2.0f*M_PI*r1;
    uphi->sinphi = sin(phi);
    uphi->cosphi = cos(phi);
    
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

void accumEndep() {
    
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    /* Accumulate endep and endep squared for statistical analysis */
    double edep = 0.0;
    
    for (int irl=0; irl<gridsize + 1; irl++) {
        edep = score.endep[irl];
        
        score.accum_endep[irl] += edep;
        score.accum_endep2[irl] += pow(edep, 2.0);
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
            
    for (int iz=0; iz<geometry.ksize; iz++) {
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
                    unc_endep = endep2 - pow(endep, 2.0);
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
    for (int iz=0; iz<geometry.ksize; iz++) {
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
    
    //Accumulate the results
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
    char buffer[128];
    
    if (getInputValue(buffer, "output folder") != 1) {
        printf("Can not find 'output folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(output_folder, buffer);
    
    /* Make space for the new string */
    char* file_name = malloc(strlen(output_folder) + strlen(output_file) + strlen(extension) + 1);
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
    
    fclose(fp);
    return;
}

void ausgab(double edep) {
    
    int np = stack.np;
    int irl = stack.ir[np];
    double endep = stack.wt[np]*edep;
    
    /* Deposit particle energy on spot */
    score.endep[irl] += endep;
    
    return;
}

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

void initHistory(int ibeamlet) {
    
    double rnno1;
    double rnno2;
    
    int ijmax = geometry.isize*geometry.jsize;
    int imax = geometry.isize;
    
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
     defined by the bixel at isocenter*/    
    double xiso = 0.0; 
    double yiso = 0.0;
    double ziso = 0.0;
    
    rnno1 = setRandom();
    rnno2 = setRandom();
    
    xiso = rnno1*source.xside1[ibeamlet] + rnno2*source.xside2[ibeamlet] + 
            source.xcorner[ibeamlet];
    yiso = rnno1*source.yside1[ibeamlet] + rnno2*source.yside2[ibeamlet] + 
            source.ycorner[ibeamlet];
    ziso = rnno1*source.zside1[ibeamlet] + rnno2*source.zside2[ibeamlet] + 
            source.zcorner[ibeamlet];
    
    /* Norm of the resulting vector from the source of current beam to the 
     position of the particle on bixel */
    int ibeam = source.ibeam[ibeamlet];
    double vnorm = sqrt(pow(xiso - source.xsource[ibeam], 2.0) + 
            pow(yiso - source.ysource[ibeam], 2.0) + 
            pow(ziso - source.zsource[ibeam], 2.0));
        
    /* Direction of the particle from position on bixel to beam source*/
    double u = -(xiso - source.xsource[ibeam])/vnorm;
    double v = -(yiso - source.ysource[ibeam])/vnorm;
    double w = -(ziso - source.zsource[ibeam])/vnorm;
    
    /* Calculate the minimum distance from particle position on bixel to 
     phantom boundaries */
    double ustep = 1.0E5;
    double dist;
    
    if(u > 0.0) {
        dist = (geometry.xbounds[geometry.isize]-xiso)/u;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    if(u < 0.0) {
        dist = -(xiso-geometry.xbounds[0])/u;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    
    if(v > 0.0) {
        dist = (geometry.ybounds[geometry.jsize]-yiso)/v;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    if(v < 0.0) {
        dist = -(yiso-geometry.ybounds[0])/v;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    
    if(w > 0.0) {
        dist = (geometry.zbounds[geometry.ksize]-ziso)/w;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    if(w < 0.0) {
        dist = -(ziso-geometry.zbounds[0])/w;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    
    /* Transport particle from bixel to surface. Adjust particle direction 
     to be incident to phantom surface */
    stack.x[stack.np] = xiso + ustep*u;
    stack.y[stack.np] = yiso + ustep*v;
    stack.z[stack.np] = ziso + ustep*w;
    
    stack.u[stack.np] = -u;
    stack.v[stack.np] = -v;
    stack.w[stack.np] = -w;
    
    /* Determine region index of source particle */
    int ix = 0;
    while ((geometry.xbounds[ix+1] < stack.x[stack.np])) {
        ix++;
    }
    
    int iy = 0;
    while ((geometry.ybounds[iy+1] < stack.y[stack.np])) {
        iy++;
    }
    
    int iz = 0;
    while ((geometry.xbounds[iz+1] < stack.z[stack.np])) {
        iz++;
    }
    
    stack.ir[stack.np] = 1 + ix + iy*imax + iz*ijmax;
          
    /* Set statistical weight and distance to closest boundary*/
    stack.wt[stack.np] = 1.0;
    stack.dnear[stack.np] = 0.0;
    
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

void initMediaData(){
    
    /* Array that indicates if medium was found on pegs file or not */
    int *media_found = (int*) malloc(geometry.nmed*sizeof(int));
    
    /* Get media information from pegs file */
    int nmedia_found;
    nmedia_found = readPegsFile(media_found);
    
    /* Check if all requested media was found */
    if (nmedia_found < geometry.nmed) {
        printf("The following media were not found or could not be read "
               "from pegs file:\n");
        for (int i=0; i<geometry.nmed; i++) {
            if (media_found[i] == 0) {
                printf("\t %s\n", geometry.med_names[i]);
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
    char pegs_file[128];
    char buffer[128];
    
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
    electron_data.blcc = malloc(geometry.nmed*sizeof(double));
    electron_data.xcc = malloc(geometry.nmed*sizeof(double));
    electron_data.eke0 = malloc(geometry.nmed*sizeof(double));
    electron_data.eke1 = malloc(geometry.nmed*sizeof(double));
    electron_data.esig0 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.esig1 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.psig0 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.psig1 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.ededx0 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.ededx1 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.pdedx0 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.pdedx1 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.ebr10 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.ebr11 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.pbr10 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.pbr11 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.pbr20 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.pbr21 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.tmxs0 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron_data.tmxs1 = malloc(geometry.nmed*MXEKE*sizeof(double));
    
    /* Zero the following arrays, as they are surely not totally used. */
    memset(electron_data.esig0, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.esig1, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.psig0, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.psig1, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.ededx0, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.ededx1, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.pdedx0, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.pdedx1, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.ebr10, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.ebr11, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.pbr10, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.pbr11, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.pbr20, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.pbr21, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.tmxs0, 0.0, geometry.nmed*MXEKE*sizeof(double));
    memset(electron_data.tmxs1, 0.0, geometry.nmed*MXEKE*sizeof(double));
    
    do {
        /* Read a line of pegs file */
        char buffer[80];
        fgets(buffer, sizeof(buffer), fp);
        
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
            
            for (int i = 0; i < geometry.nmed; i++) {
                char cname[20];
                strncpy(cname, geometry.med_names[i], 20);
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
            strncpy(pegs_data.names[imed], name, 60);
            pegs_data.ne[imed] = 0;
            
            /* Read the next line containing the density, number of elements
             and flags */
            fgets(buffer, 80, fp);
            int ok = 1;
            char s[100];
            char s2[100];
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
                fgets(buffer, 80, fp);
                char s[100];
                char s2[100];
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
            fgets(buffer, 80, fp);
            
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
            fgets(buffer, 80, fp);
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
                fgets(buffer, 100, fp);
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
    } while ((nmedia < geometry.nmed) && !feof(fp));
    
    /* Print some information for debugging purposes */
    if(verbose_flag) {
        for (int i=0; i<geometry.nmed; i++) {
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

void initPhotonData() {
    
    /* Get file path from input data */
    char photon_xsection[128];
    char buffer[128];
    
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
    
    photon_data.ge0 = malloc(geometry.nmed*sizeof(double));
    photon_data.ge1 = malloc(geometry.nmed*sizeof(double));
    photon_data.gmfp0 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon_data.gmfp1 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon_data.gbr10 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon_data.gbr11 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon_data.gbr20 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon_data.gbr21 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon_data.cohe0 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon_data.cohe1 = malloc(geometry.nmed*MXGE*sizeof(double));
    
    for (int i=0; i<geometry.nmed; i++) {
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

void cleanPhoton() {
    
    free(photon_data.ge0);
    free(photon_data.ge1);
    free(photon_data.gmfp0);
    free(photon_data.gmfp1);
    free(photon_data.gbr10);
    free(photon_data.gbr11);
    free(photon_data.gbr20);
    free(photon_data.cohe0);
    free(photon_data.cohe1);
    
    return;
}

void listPhoton() {
    
    /* Get file path from input data */
    char output_folder[128];
    char buffer[128];
    
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
    for (int i=0; i<geometry.nmed; i++) {
        fprintf(fp, "For medium %s: \n", geometry.med_names[i]);
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

void initRayleighData(void) {
    
    double *xval = (double*) malloc(MXRAYFF*sizeof(double));
    double **aff = (double**) malloc(MXELEMENT*sizeof(double*));
    double *ff = malloc(MXRAYFF*geometry.nmed*sizeof(double));
    double *pe_array = malloc(MXGE*geometry.nmed*sizeof(double));

    for (int i=0; i<MXELEMENT; i++) {
        aff[i] = malloc(MXRAYFF*sizeof(double));
    }
    
    /* Read Rayleigh atomic form factor from pgs4form.dat file */
    readFfData(xval, aff);
    
    /* Allocate memory for Rayleigh data */
    rayleigh_data.xgrid = malloc(geometry.nmed*MXRAYFF*sizeof(double));
    rayleigh_data.fcum = malloc(geometry.nmed*MXRAYFF*sizeof(double));
    rayleigh_data.b_array = malloc(geometry.nmed*MXRAYFF*sizeof(double));
    rayleigh_data.c_array = malloc(geometry.nmed*MXRAYFF*sizeof(double));
    rayleigh_data.i_array = malloc(geometry.nmed*RAYCDFSIZE*sizeof(int));
    rayleigh_data.pmax0 = malloc(geometry.nmed*MXGE*sizeof(double));
    rayleigh_data.pmax1 = malloc(geometry.nmed*MXGE*sizeof(double));
    
    for (int i=0; i<geometry.nmed; i++) {
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

void readFfData(double *xval, double **aff) {
    
    /* Get file path from input data */
    char pgs4form_file[128];
    char buffer[128];
    
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

void cleanRayleigh() {
    
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
    char buffer[128];
    
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
    for (int i=0; i<geometry.nmed; i++) {
        fprintf(fp, "For medium %s: \n", geometry.med_names[i]);
        
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

void initPairData() {
    /* The data calculated here corresponds partially to the subroutine
     fix_brems in egsnrc.macros. This subroutine calculates the parameters
     for the rejection function used in bremsstrahlung sampling*/

    double Zf, Zb, Zt, Zg, Zv; // medium functions, used for delx and delcm
    double fmax1, fmax2;
    
    /* Memory allocation */
    pair_data.dl1 = malloc(geometry.nmed*8*sizeof(double));
    pair_data.dl2 = malloc(geometry.nmed*8*sizeof(double));
    pair_data.dl3 = malloc(geometry.nmed*8*sizeof(double));
    pair_data.dl4 = malloc(geometry.nmed*8*sizeof(double));
    pair_data.dl5 = malloc(geometry.nmed*8*sizeof(double));
    pair_data.dl6 = malloc(geometry.nmed*8*sizeof(double));
    
    pair_data.bpar0 = malloc(geometry.nmed*sizeof(double));
    pair_data.bpar1 = malloc(geometry.nmed*sizeof(double));
    pair_data.delcm = malloc(geometry.nmed*sizeof(double));
    pair_data.zbrang = malloc(geometry.nmed*sizeof(double));
    
    int nmed = geometry.nmed;
    
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
    char buffer[128];
    
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
    for (int i=0; i<geometry.nmed; i++) {
        fprintf(fp, "For medium %s: \n", geometry.med_names[i]);
        
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

void photon() {
    
    int np = stack.np;              // stack pointer
    int irl = stack.ir[np];         // region index
    int imed = region.med[irl];     // medium index of current region
    double edep = 0.0;              // deposited energy by particle
    double eig = stack.e[np];       // energy of incident gamma
    
    /* First check for photon cutoff energy */
    if (eig <= region.pcut[irl]) {
        edep = eig;
        
        /* Deposit energy on the spot */
        ausgab(edep);
        stack.np -= 1;
        return;
    }
    
    /* Select photon mean free path */
    double rnno = setRandom();
    if (rnno < 1.0E-30) {
        rnno = 1.0E-30;
    }
    double dpmfp = (-1.0)*log(rnno);
    
    int lgle = 0;           // index for gamma MFP interpolation
    double gle = log(eig);  /* gle is gamma log energy, here to sample number
                             of mfp to transport before interacting */
    double cohfac = 0.0;    // Rayleigh scattering correction
    int ptrans = 1;         // variable to control photon transport (true)
    int pmedium = 1;        /* variable to control pass of photon to
                             another medium (true) */
    
    do {    /* start of "new medium" loop */
        double gmfpr0 = 0.0;  /* photon mfp before density and coherent
                               correction */
        
        if (imed != -1) {
            /* Adjust lgle to C indexing */
            lgle = pwlfInterval(imed, gle,
                                photon_data.ge1, photon_data.ge0) - 1;
            gmfpr0 = pwlfEval(imed*MXGE + lgle, gle,
                              photon_data.gmfp1, photon_data.gmfp0);
            
        }
        
        do {    /* start of "photon transport loop */
            double tstep;       // distance to a discrete interaction
            double gmfp = 0.0;  // photon MFP after density scaling
            double rhof;        // mass density ratio
            
            if (imed == -1) {
                tstep = 10.0E8; // i.e. infinity
            }
            else {
                /* Density ratio scaling templates */
                rhof = region.rhof[irl];
                gmfp = gmfpr0/rhof;
                
                /* Rayleigh scattering template */
                cohfac = pwlfEval(imed*MXGE + lgle, gle,
                                  photon_data.cohe1, photon_data.cohe0);
                gmfp *= cohfac;
                tstep = gmfp*dpmfp;
            }

            int irnew = irl;       // default new region number
            int idisc = 0;          // assume photon is not discarded
            double ustep = tstep;    // transfer transport distance to user variable

            howfar(&idisc, &irnew, &ustep);
            
            if (idisc > 0) {
                /* User requested inmediate discard */
                edep = eig;
                ausgab(edep);
                stack.np -= 1;
                return;
            }
            
            /* Transport distance after truncation by howfar */
            double vstep = ustep;
            
            /* Transport the photon */
            stack.x[np] += vstep*stack.u[np];
            stack.y[np] += vstep*stack.v[np];
            stack.z[np] += vstep*stack.w[np];
            
            if (imed != -1) {
                /* Deduct mean free path */
                dpmfp = fmax(0.0, dpmfp-ustep/gmfp);
            }
            int irold = irl;    /* save region before transport */
            int medold = imed;
            
            if (irnew != irold) {
                /* Region change */
                stack.ir[np] = irnew;
                irl = stack.ir[np];
                imed = region.med[irl];
            }

            if (eig <= region.pcut[irl]) {
                edep = eig;
                
                /* Deposit energy on the spot */
                ausgab(edep);
                stack.np -= 1;
                return;
            }
            
            if (imed != medold) {
                ptrans = 0;
            }
            
            if (imed != -1 && dpmfp <= SGMFP) {
                /* Time for an interaction */
                pmedium = 0;
                ptrans = 0;
            }

        } while (ptrans);   /* end of "photon transport loop */
    } while (pmedium); /* end of "new medium" loop */
    
    /* Time for an interaction. First check for Rayleigh scattering */
    rnno = setRandom();
    if (rnno <= 1.0 - cohfac) {
        /* It was Rayleigh */
        rayleigh(imed, eig, gle, lgle);
    }
    else {
        /* Other interactions */
        rnno = setRandom();
        
        /* gbr1 = pair/(pair + compton + photo) = pair/gtotal */
        double gbr1 = pwlfEval(imed*MXGE + lgle, gle,
                               photon_data.gbr11, photon_data.gbr10);
        if (rnno <= gbr1 && eig>2.0*RM) {
            /* It was pair production */
            pair(imed);
            
            np = stack.np;
            if (stack.iq[np] != 0) {
                /* Electron to be transported next */
                return;
            }
        }
        
        /* gbr2 = (pair + compton)/gtotal */
        double gbr2 = pwlfEval(imed*MXGE + lgle, gle,
                               photon_data.gbr21, photon_data.gbr20);
        
        if (rnno < gbr2) {
            /* It was compton */
            compton();
            
            np = stack.np;
            if (stack.iq[np] != 0) {
                /* Electron to be transported next */
                return;
            }
        }
        else {
            /* It was photoelectric */
            photo();
            
            if (stack.iq[np] != 0) {
                /* Electron to be transported next */
                return;
            }
        }
    }
    
    /* If here, photon is the lowest energy particle. The particle exits
     this function and re-enters to resume transport process */
    
    return;
}

void rayleigh(int imed, double eig, double gle, int lgle) {
    
    int ibin, ib;
    double xv, costhe, csqthe, sinthe;
    double rnno0, rnno1;
    double pmax = pwlfEval(imed*MXGE + lgle, gle,
                           rayleigh_data.pmax1, rayleigh_data.pmax0);
    double xmax = HC_INVERSE*eig;
    double dwi = (double)RAYCDFSIZE - 1.0;
    
    do {
        rnno1 = setRandom();
        
        do {
            rnno0 = setRandom();
            rnno0 *= pmax;
            
            /* For the following indexes the C convention must be used */
            ibin = (int)rnno0*dwi;
            ib = rayleigh_data.i_array[ibin] - 1;
            
            if((rayleigh_data.i_array[ibin+1] - 1) > ib) {
                while(rnno0 >= rayleigh_data.fcum[ib+1]) {
                    ib++;
                }
            }
            
            rnno0 = (rnno0 - rayleigh_data.fcum[ib])*rayleigh_data.c_array[ib];
            xv = rayleigh_data.xgrid[ib]*exp(log(1.0 + rnno0)*
                                             rayleigh_data.b_array[ib]);
        } while(xv >= xmax);
        
        xv /= eig;
        costhe = 1.0 - TWICE_HC2*pow(xv, 2.0);
        csqthe = pow(costhe, 2.0);
    } while(2.0*rnno1 >= (1.0 + csqthe));
    
    sinthe = sqrt(1.0 - csqthe);
    
    struct Uphi uphi;
    uphi21(&uphi, costhe, sinthe);
    
    return;
}

double setPairRejectionFunction(int imed, double xi, double esedei,
                                double eseder, double tteig) {
    
    double rej = 2.0 + 3.0*(esedei + eseder) - 4.0*(esedei + eseder +
        1.0 - 4.0*pow((xi - 0.5), 2.0))*(1.0 +
            0.25*log((pow(((1.0 + eseder)*(1.0 + esedei)/(2.0*tteig)),2.0)) +
                pair_data.zbrang[imed]*pow(xi, 2.0)));
    
    return rej;
}

void pair(int imed) {
    
    int np = stack.np;
    double eig = stack.e[np];   /* energy of incident photon */
    
    double ese1, ese2;          /* energy of "electrons" */
    int iq1, iq2;               /* charge of "electrons" */
    int l, l1;                  /* flags for high/low energy distributions */
    
    if (eig <= 2.1) {
        /* Use low energy pair production aproximation. It corresponds to an
         uniform energy distribution */
        double rnno0 = setRandom();
        double rnno1 = setRandom();
        
        ese2 = RM + 0.5*rnno0*(eig - 2.0*RM);
        ese1 = eig - ese2;
        
        if (rnno1 < 0.5) {
            iq1 = -1;
            iq2 = 1;
        } else {
            iq1 = 1;
            iq2 = -1;
        }
        
    } else {
        /* Must sample. Now decide whether to use Bethe-Heitler or BH
         Coulomb corrected */
        double Amax;    /* maximum of the screening function used with
                         (br-1/2)^2 */
        double Bmax;    /* maximum of the screening function used with the
                         uniform part */
        double delta;   /* scaled momentum transfer */
        double aux;
        
        if (eig < 50.0) {
            /* Use BH without Coulomb correction */
            l = 4;
            l1 = l + 1;
            
            /* Find the actual rejection maximum for this photon energy */
            delta = 4.0*pair_data.delcm[imed]/eig;
            if(delta < 1.0) {
                Amax = pair_data.dl1[imed*8+l] +
                    delta*(pair_data.dl2[imed*8+l] +
                        delta*pair_data.dl3[imed*8+l]);
                Bmax = pair_data.dl1[imed*8+l1] +
                    delta*(pair_data.dl2[imed*8+l1] +
                        delta*pair_data.dl3[imed*8+l1]);
            }
            else {
                aux = log(delta + pair_data.dl6[imed*8+l]);
                Amax = pair_data.dl4[imed*8+l] + pair_data.dl5[imed*8+l]*aux;
                Bmax = pair_data.dl4[imed*8+l1] + pair_data.dl5[imed*8+l1]*aux;
            }
            /* and then calculate the probability for sampling from (br-1/2)^2*/
            aux = 1.0 - 2.0*RM/eig;
            aux = pow(aux, 2.0);
            aux *= Amax/3.0;
            aux /= (Bmax + aux);

        } else {
            /* Use BH Coulomb-corrected */
            l = 6;
            
            /* The absolute maxima are close to the actual maxima
             at high energies */
            Amax = pair_data.dl1[imed*8+l];
            Bmax = pair_data.dl1[imed*8+l+1];
            aux = pair_data.bpar1[imed]*(1.0 - pair_data.bpar0[imed]*RM/eig);

        }
        
        double eavail = eig - 2.0*RM;    /* energy available after
                                         pair production */
        double rejf;     /* screening rejection function */
        double rejmax;   /* the maximum of rejmax */
        double rnno4;

        do {
            double br; /* fraction of the available energy (eig-2*RM) going to
                       the lower energy electron */
            double rnno0 = setRandom();
            double rnno1 = setRandom();
            rnno4 = setRandom();
            
            if(rnno0 > aux) {
                /* Use the uniform part of the distribution */
                br = 0.5*rnno1;
                rejmax = Bmax;
                l1 = l + 1;
            }
            else {
                /* Use the (br-1/2)^2 part */
                double rnno2 = setRandom();
                double rnno3 = setRandom();
                br = 0.5*(1.0 - fmax(fmax(rnno1, rnno2), rnno3));
                rejmax = Amax;
                l1 = l;
            }
            
            ese2 = br*eavail + RM;
            ese1 = eig - ese2;
            delta = (eig*pair_data.delcm[imed])/(ese2*ese1);
            if(delta < 1.0) {
                rejf = pair_data.dl1[imed*8+l1] +
                    delta*(pair_data.dl2[imed*8+l1] +
                           delta*pair_data.dl3[imed*8+l1]);
            }
            else {
                rejf = pair_data.dl4[imed*8+l1] +
                    pair_data.dl5[imed*8+l1]*
                        log(delta + pair_data.dl6[imed*8+l1]);
            }
            
        } while(rnno4*rejmax > rejf);
        
        ese1 = eig - ese2;
        rnno4 = setRandom();
        
        if (rnno4 < 0.5) {
            iq1 = -1;
            iq2 = 1;
        } else {
            iq1 = 1;
            iq2 = -1;
        }
    }
    
    /* Energy going to lower secondary has now been determined */
    stack.e[np] = ese1;
    stack.e[np+1] = ese2;
    
    /* Set pair angle and direction of charged particles. The angle is selected
     from the leading term of the angular distribution */
    double ese;
    double sinthe;
    double costhe;
    struct Uphi uphi;

    for(int j=0; j<2; j++) {
        /* This loop is executed twice, once for each generated particle. The
         following corresponds to iprdst == 2 (MOTZ, OLSEN AND KOCH 1969) */
        if(j == 0) {
            ese = ese1;
        }
        else {
            ese = ese2;
        }
        
        double tteig = eig/RM;  /* initial photon energy in electron RM units */
        double ttese = ese/RM;  /* final electron energy in electron RM units */
        
        /* Ratio of particle's energies (r in PIRS0287) */
        double esedei = ttese/(tteig - ttese);
        double eseder = 1.0/esedei;
        
        /* Determine the normalization */
        double ximin = 1.0/(1.0 + pow((M_PI*ttese), 2.0));
        
        /* Set pair rejection function eq. 4 PIRS 0287 */
        double rejmin = setPairRejectionFunction(imed, ximin, esedei,
                                                 eseder, tteig);
        
        double ya = pow((2.0/tteig), 2.0);
        double xitry = fmax(0.01, fmax(ximin, fmin(0.5,
            sqrt(ya/pair_data.zbrang[imed]))));
        double galpha = 1.0 + 0.25*log(ya +
            pair_data.zbrang[imed]*pow(xitry, 2.0));
        double gbeta = 0.5*pair_data.zbrang[imed]*xitry/(ya +
            pair_data.zbrang[imed]*pow(xitry, 2.0));
        galpha -= gbeta*(xitry - 0.5);
        double ximid = galpha/(3.0*gbeta);
        
        if (galpha >= 0.0) {
            ximid = 0.5 - ximid + sqrt(pow(ximid, 2.0) + 0.25);
        } else{
            ximid = 0.5 - ximid - sqrt(pow(ximid, 2.0) + 0.25);
        }
        
        ximid = fmax(0.01, fmax(ximin, fmin(0.5, ximid)));
        
        /* Set pair rejection function eq. 4 PIRS 0287 */
        double rejmid = setPairRejectionFunction(imed, ximid, esedei,
                                                 eseder, tteig);
        
        /* Estimate maximum of the rejection function for later use by the
         rejection technique */
        double rejtop = 1.0*fmax(rejmin, rejmid);
        
        double theta;
        double rejtst;
        double rejfactor;
        double rtest;
        
        do {
            double xitst = setRandom();
            
            /* Set pair rejection function eq. 4 PIRS 0287 */
            rejtst = setPairRejectionFunction(imed, xitst, esedei,
                                              eseder, tteig);
            
            rtest = setRandom();
            
            /* Convert the successful candidate xitst to an angle */
            theta = sqrt(1.0/xitst - 1.0)/ttese;
            
            /* Loop until rejection technique accepts xitst */
            rejfactor = rejtst/rejtop;
        } while((rtest > rejfactor) && (theta >= M_PI));
        
        sinthe = sin(theta);
        costhe = cos(theta);
        
        if (j == 0) {
            /* First Particle */
            uphi21(&uphi, costhe, sinthe);
        }
        else {
            /* Second Particle */
            sinthe = -sinthe;
            
            /* Update stack index, it is needed by uphi32() */
            np += 1;
            stack.np = np;
            uphi32(&uphi, costhe, sinthe);
        }
    }

    /* Assign charge to new particles. The stack index was already updated,
     therefore the new particles correspond to np and np-1 indices */
    stack.iq[np] = iq2;
    stack.iq[np-1] = iq1;

    return;
}

void compton() {
    
    int np = stack.np;
    double eig = stack.e[np];
    double ko = stack.e[np]/RM;
    double broi = 1.0 + 2.0*ko;
    double bro = 1.0/broi;
    
    /* Sampling of the Klein-Nishina scattering angle */
    int first_time = 1; /* i.e. true */
    double sinthe = 0.0;
    double costhe = 0.0;
    double br;  /* scattered photon energy fraction */
    
    double rnno1, rnno2, rnno3;
    
    double aux;
    double rejf3;   /* rejection function */
    double temp;    /* aux. variable for polar angle calculation */
    
    double alph1 = 0.0;    /* probability for the 1/br part */
    double alph2 = 0.0;    /* probability for the br part */
    double alpha = 0.0;
    double rejmax = 0.0;   /* max. of rejf3 in the case of uniform sampling */

    do {
        if(ko > 2.0) {
            /* At high energies the original EGS4 method is more efficient */
            if (first_time){
                alph1 = log(broi);
                alph2 = ko*(broi + 1.0)*pow(bro, 2.0);
                alpha = alph1 + alph2;
            }
            do {
                rnno1 = setRandom();
                rnno2 = setRandom();
                
                if(rnno1*alpha < alph1) {
                    /* use 1/br part */
                    br = exp(alph1*rnno2)*bro;
                }
                else {
                    /* use the br part */
                    br = sqrt(rnno2*pow(broi, 2.0) + (1.0 - rnno2))*bro;
                }
                
                temp = (1.0 - br)/(ko*br);
                sinthe = fmax(0.0, temp*(2.0 - temp));
                aux = 1.0 + pow(br, 2.0);
                rejf3 = aux - br*sinthe;
                
                rnno3 = setRandom();
            } while(rnno3*aux > rejf3);
        }
        else {
            /* At low energies it is faster to sample br uniformely */
            if(first_time) {
                rejmax = broi + bro;
            }
            do {
                rnno1 = setRandom();
                rnno2 = setRandom();
                
                br = bro + (1.0 - bro)*rnno1;
                temp = (1.0 - br)/(ko*br);
                sinthe = fmax(0.0, temp*(2.0 - temp));
                
                rejf3 = 1.0 + pow(br, 2.0) - br*sinthe;
            } while(rnno2*br*rejmax > rejf3);
        }
        
        first_time = 0; /* i.e. false */
    } while((br < bro) || (br > 1));

    costhe = 1.0 - temp;
    sinthe = sqrt(sinthe);
    
    double esg = br*eig;            /* new energy of the photon */
    double ese = eig - esg + RM;    /* energy of the electron */
    stack.e[np] = esg;  /* change of energy */
    
    /* Adjust direction of photon */
    struct Uphi uphi;
    uphi21(&uphi, costhe, sinthe);
    
    /* Adding new electron to stack. Update stack counter for uphi32() */
    np += 1;
    stack.np = np;
    
    /* Adjust direction of new electron */
    aux = 1.0 + pow(br, 2.0) - 2.0*br*costhe;
    if(aux > 1.0E-8) {
        costhe = (1.0 - br*costhe)/sqrt(aux);
        sinthe = (1.0 - costhe)*(1.0 + costhe);
        if(sinthe > 0.0) {
            sinthe = -sqrt(sinthe);
        }
        else {
            sinthe = 0.0;
        }
    }
    else {
        costhe = 0.0;
        sinthe = -1.0;
    }
    
    uphi32(&uphi, costhe, sinthe);
    stack.e[np] = ese;
    stack.iq[np] = -1;
    
    return;
}

void photo() {
    
    int np = stack.np;
    
    /* Set energy and charge of the new electron */
    stack.e[np] += RM;
    stack.iq[np] = -1;
    
    /* Now sample photo-electron direction */
    double eelec = stack.e[np];
    
    if (eelec > region.ecut[stack.ir[np]]){
        /* Velocity of electron in c units */
        double beta = sqrt((eelec - RM)*(eelec + RM))/eelec;
        
        double costhe;
        double sinthe;
        double sinth2;
        double gamma = eelec/RM;
        double alpha = 0.5*gamma - 0.5 + 1.0/gamma; /* kinematic factor */
        double ratio = beta/alpha;
        double rnpht2;
        double xi;
        
        do {
            double rnpht = setRandom();
            rnpht = 2.0*rnpht - 1.0;
            if (ratio <= 0.2){
                double fkappa = rnpht + 0.5*ratio*(1.0 - rnpht)*(1.0 + rnpht);
                if (gamma < 100.0) {
                    costhe = (beta + fkappa)/(1.0 + beta*fkappa);
                }
                else {
                    if (fkappa > 0.0) {
                        costhe = 1.0 - (1.0 - fkappa)*(gamma - 3.0)/
                            (2.0*(1.0 + fkappa)*pow((gamma-1.0), 3.0));
                    }
                    else {
                        costhe = (beta + fkappa)/(1.0 + beta*fkappa);
                    }
                }
                xi = (1.0 + beta*fkappa)*pow(gamma, 2.0);
            }
            else {
                xi = pow(gamma, 2.0)*(1.0 + alpha*(sqrt(1.0f
                        + ratio*(2.0*rnpht + ratio)) - 1.0));
                costhe = (1.0 - 1.0/xi)/beta;
            }
            sinth2 = fmax((1.0 - costhe)*(1.0 + costhe), 0.0);
            rnpht2 = setRandom();
        } while(rnpht2 > 0.5*(1.0 + gamma)*sinth2*xi/gamma);
        
        sinthe = sqrt(sinth2);
        
        struct Uphi uphi;
        uphi21(&uphi, costhe, sinthe);
    }
    
    return;
}

void initMscatData() {
    
    int nmed = geometry.nmed;
    
    readRutherfordMscat(nmed);
    
    
    for (int imed=0; imed<nmed; imed++) {
        /* Absorb Euler constant into the multiple scattering parameter */
        electron_data.blcc[imed] = 1.16699413758864573*electron_data.blcc[imed];
        
        /* Take its square as this is employed throughout */
        electron_data.xcc[imed] = pow(electron_data.xcc[imed], 2.0);
    }
    
    /* Initialize data for spin effects */
    initSpinData(nmed);
    
    /* Determine maximum cross-section per energy loss for every medium */
    double esige_max = 0.0;
    double psige_max = 0.0;
    double sigee, sigep, sige_old, sigp_old, p2, beta2, chi_a2, si, sip1;
    double ei, eil, ededx, dedx0, sig, eip1, eke, elke, ekef, elkef,
        aux, estepx, eip1l, elktmp, ektmp;
    int neke, ise_monoton, isp_monoton, leil, lelke, lelkef, leip1l, lelktmp;
    
    /* Allocate memory for electron data */
    electron_data.sig_ismonotone = malloc(2*nmed*sizeof(int));
    electron_data.esig_e = malloc(nmed*sizeof(double));
    electron_data.psig_e = malloc(nmed*sizeof(double));
    electron_data.e_array = malloc(nmed*MXEKE*sizeof(double));
    electron_data.range_ep = malloc(2*nmed*MXEKE*sizeof(double));
    electron_data.expeke1 = malloc(nmed*sizeof(double));
    
    /* Zero the following arrays, as they surely are not totally used. */
    memset(electron_data.e_array, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.range_ep, 0.0, 2*nmed*MXEKE*sizeof(double));

    for (int imed=0; imed<nmed; imed++) {
        sigee = 1.0E-15;
        sigep = 1.0E-15;
        neke = pegs_data.meke[imed]; /* Number of elements in storage array */
        
        /* The following are boolean variables, both true by default */
        ise_monoton = 1;
        isp_monoton = 1;
        
        sige_old = -1.0;
        sigp_old = -1.0;
        
        for (int i=1; i<=neke; i++) {
            ei   = exp(((double)i - electron_data.eke0[imed])/electron_data.eke1[imed]);
            eil  = log(ei);
            leil = i - 1; /* Consider C indexing */
            
            ededx = electron_data.ededx1[imed*MXEKE + leil]*eil +
                electron_data.ededx0[imed*MXEKE + leil];
            sig = electron_data.esig1[imed*MXEKE + leil]*eil +
                electron_data.esig0[imed*MXEKE + leil];
            
            sig /= ededx;
            if (sig > sigee) {
                sigee = sig;
            }
            if (sig < sige_old) {
                ise_monoton = 0; /* i.e. false */
            }
            sige_old = sig;
            
            ededx = electron_data.pdedx1[imed*MXEKE + leil]*eil +
                electron_data.pdedx0[imed*MXEKE + leil];
            sig = electron_data.psig1[imed*MXEKE + leil]*eil +
                electron_data.psig0[imed*MXEKE + leil];
            
            sig /= ededx;
            if (sig>sigep) {
                sigep = sig;
            }
            if (sig<sigp_old) {
                isp_monoton = 0; /* i.e. false */
            }
            sigp_old = sig;

        }
        electron_data.sig_ismonotone[0*nmed + imed] = ise_monoton;
        electron_data.sig_ismonotone[1*nmed + imed] = isp_monoton;
        electron_data.esig_e[imed] = sigee;
        electron_data.psig_e[imed] = sigep;
        
        if (sigee > esige_max) {
            esige_max = sigee;
        }
        if (sigep > psige_max) {
            psige_max = sigep;
        }
    }
    
    /* Determine upper limit in step size for multiple scattering */
    for (int imed=0; imed<nmed; imed++) {
        /* Calculate range arrays first */
        
        /* Energy of first table entry */
        ei = exp((1.0 - electron_data.eke0[imed])/electron_data.eke1[imed]);
        eil = log(ei);
        leil = 0;
        electron_data.e_array[imed*MXEKE] = ei;
        electron_data.expeke1[imed] = exp(1.0/electron_data.eke1[imed]) - 1.0;
        electron_data.range_ep[0*nmed*MXEKE + imed*MXEKE] = 0.0;
        electron_data.range_ep[1*nmed*MXEKE + imed*MXEKE] = 0.0;
        
        neke = pegs_data.meke[imed]; /* number of elements in storage array */
        for (int i=1; i<=neke-1; i++) {
            /* Energy at i + 1*/
            eip1 = exp(((double)(i + 1) -
                        electron_data.eke0[imed])/electron_data.eke1[imed]);
            electron_data.e_array[imed*MXEKE + i] = eip1;
            
            /* Calculate range. The following expressions result from the
             logarithmic interpolation for the (restricted) stopping power
             and a power series expansion of the integral */
            eke = 0.5*(eip1 + ei);
            elke = log(eke);
            lelke = (int)(electron_data.eke1[imed]*elke +
                          electron_data.eke0[imed]) - 1;
            ededx = electron_data.pdedx1[imed*MXEKE + lelke]*elke +
                electron_data.pdedx0[imed*MXEKE + lelke];
            aux = electron_data.pdedx1[imed*MXEKE + i - 1]/ededx;
            
            electron_data.range_ep[1*nmed*MXEKE + imed*MXEKE + i] =
                electron_data.range_ep[1*nmed*MXEKE + imed*MXEKE + i - 1] +
                    (eip1-ei)/ededx*(1.0 +
                        aux*(1.0 + 2.0*aux)*pow((eip1-ei)/eke, 2.0)/24.0);
            
            ededx = electron_data.ededx1[imed*MXEKE + lelke]*elke +
                electron_data.ededx0[imed*MXEKE + lelke];
            aux = electron_data.ededx1[imed*MXEKE + i - 1]/ededx;
            
            electron_data.range_ep[0*nmed*MXEKE + imed*MXEKE + i] =
                electron_data.range_ep[0*nmed*MXEKE + imed*MXEKE + i - 1] +
                    (eip1 - ei)/ededx*(1.0 + aux*(1.0 + 2.0*aux)*
                        pow(((eip1 - ei)/eke), 2.0)/24.0);
            
            ei = eip1;
        }
        
        /* Now tmxs */
        eil = (1.0 - electron_data.eke0[imed])/electron_data.eke1[imed];
        ei  = exp(eil); leil = 1;
        p2  = ei*(ei + 2.0*RM);
        beta2 = p2/(p2 + pow(RM, 2.0));
        chi_a2 = electron_data.xcc[imed]/(4.0*p2*electron_data.blcc[imed]);
        dedx0 = electron_data.ededx1[imed*MXEKE + leil]*eil +
            electron_data.ededx0[imed*MXEKE + leil];
        estepx = 2.0*p2*beta2*dedx0/ei/electron_data.xcc[imed]/
            (log(1.0 + 1.0/chi_a2)*(1.0 + chi_a2) - 1.0);
        estepx *= XIMAX;
        if (estepx > ESTEPE) {
            estepx = ESTEPE;
        }
        si = estepx*ei/dedx0;
        
        for (int i=1; i<=neke - 1; i++){
            elke = ((double)(i + 1) -
                    electron_data.eke0[imed])/electron_data.eke1[imed];
            eke  = exp(elke);
            lelke = i;
            
            p2 = eke*(eke + 2.0*RM);
            beta2 = p2/(p2 + pow(RM, 2.0));
            chi_a2 = electron_data.xcc[imed]/(4.0*p2*electron_data.blcc[imed]);
            ededx = electron_data.ededx1[imed*MXEKE + lelke]*elke +
                electron_data.ededx0[imed*MXEKE + lelke];
            estepx = 2.0*p2*beta2*ededx/eke/(electron_data.xcc[imed])/
            (log(1.0 + 1.0/chi_a2)*(1.0 + chi_a2) - 1.0);
            estepx = estepx*XIMAX;
            
            if (estepx > ESTEPE) {
                estepx = ESTEPE;
            }
            ekef = (1.0 - estepx)*eke;
            if (ekef <= electron_data.e_array[imed*MXEKE]) {
                sip1 = (electron_data.e_array[imed*MXEKE] - ekef)/dedx0;
                ekef = electron_data.e_array[imed*MXEKE];
                elkef = (1.0 -
                         electron_data.eke0[imed])/electron_data.eke1[imed];
                lelkef = 0;
            }
            else{
                elkef = log(ekef);
                lelkef = electron_data.eke1[imed]*elkef +
                    electron_data.eke0[imed] - 1;
                leip1l = lelkef + 1;
                /* The value of leip1l must be adjusted by one in the following
                 sentence, due to the use of C-indexing convention */
                eip1l  = ((double)(leip1l + 1) -
                          electron_data.eke0[imed])/electron_data.eke1[imed];
                eip1   = electron_data.e_array[imed*MXEKE + leip1l];
                aux    = (eip1 - ekef)/eip1;
                elktmp = 0.5*(elkef + eip1l + 0.25*aux*aux*(1.0 +
                                                    aux*(1.0 + 0.875*aux)));
                ektmp  = 0.5*(ekef+eip1);
                lelktmp = lelkef;
                ededx = electron_data.ededx1[imed*MXEKE + lelktmp]*elktmp +
                    electron_data.ededx0[imed*MXEKE + lelktmp];
                aux = electron_data.ededx1[imed*MXEKE + lelktmp]/ededx;
                sip1 = (eip1 - ekef)/ededx*(1.0 + aux*(1.0 + 2.0*aux)*
                            (pow(((eip1-ekef)/ektmp),2.0)/24.0));
            }
            
            sip1 += electron_data.range_ep[0*nmed*MXEKE + imed*MXEKE + lelke] -
                electron_data.range_ep[0*nmed*MXEKE + imed*MXEKE + lelkef + 1];
            
            /* Now solve these equations
             si   = tmxs1 * eil   + tmxs0
             sip1 = tmxs1 * eip1l + tmxs0 */
            electron_data.tmxs1[imed*MXEKE + i - 1] =
                (sip1 - si)*electron_data.eke1[imed];
            electron_data.tmxs0[imed*MXEKE + i - 1] = sip1 -
                electron_data.tmxs1[imed*MXEKE + i - 1]*elke;
            si  = sip1;
            
        }
        electron_data.tmxs0[imed*MXEKE + neke - 1] =
            electron_data.tmxs0[imed*MXEKE + neke - 2];
        electron_data.tmxs1[imed*MXEKE + neke - 1] =
            electron_data.tmxs1[imed*MXEKE + neke - 2];
    }
    
    return;
}

void cleanElectron() {
    
    free(electron_data.blcc);
    free(electron_data.blcce0);
    free(electron_data.blcce1);
    free(electron_data.e_array);
    free(electron_data.ebr10);
    free(electron_data.ebr11);
    free(electron_data.ededx0);
    free(electron_data.ededx1);
    free(electron_data.eke0);
    free(electron_data.eke1);
    free(electron_data.esig0);
    free(electron_data.esig1);
    free(electron_data.esig_e);
    free(electron_data.etae_ms0);
    free(electron_data.etae_ms1);
    free(electron_data.etap_ms0);
    free(electron_data.etap_ms1);
    free(electron_data.expeke1);
    free(electron_data.pbr10);
    free(electron_data.pbr11);
    free(electron_data.pbr20);
    free(electron_data.pbr21);
    free(electron_data.pdedx0);
    free(electron_data.pdedx1);
    free(electron_data.psig0);
    free(electron_data.psig1);
    free(electron_data.psig_e);
    free(electron_data.q1ce_ms0);
    free(electron_data.q1ce_ms1);
    free(electron_data.q1cp_ms0);
    free(electron_data.q1cp_ms1);
    free(electron_data.q2ce_ms0);
    free(electron_data.q2ce_ms1);
    free(electron_data.q2cp_ms0);
    free(electron_data.q2cp_ms1);
    free(electron_data.range_ep);
    free(electron_data.tmxs0);
    free(electron_data.tmxs1);
    free(electron_data.xcc);
    free(electron_data.sig_ismonotone);
    
    return;
}

void listElectron(void) {
    
    /* Get file path from input data */
    char output_folder[128];
    char buffer[128];
    
    if (getInputValue(buffer, "output folder") != 1) {
        printf("Can not find 'output folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(output_folder, buffer);
    
    char file_name[256];
    strcpy(file_name, output_folder);
    strcat(file_name, "electron_data.lst");
    
    /* List electron data to output file */
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
    fprintf(fp, "Listing electron data: \n");
    for (int i=0; i<geometry.nmed; i++) {
        fprintf(fp, "For medium %s: \n", geometry.med_names[i]);
        fprintf(fp, "electron_data.blcc[%d] = %15.5f\n", i,
                electron_data.blcc[i]);
        fprintf(fp, "electron_data.xcc[%d] = %15.5f\n", i,
                electron_data.xcc[i]);
        fprintf(fp, "electron_data.eke = \n");
        fprintf(fp, "\t eke0[%d] = %15.5f, eke1[%d] = %15.5f\n", i,
                electron_data.eke0[i], i, electron_data.eke1[i]);
        fprintf(fp, "electron_data.sig_ismonotone = \n");
        fprintf(fp, "\t sig_ismonotone[0][%d] = %d, sig_ismonotone[1][%d] = %d\n", i,
                electron_data.sig_ismonotone[i*2], i, electron_data.sig_ismonotone[i*2+1]);
        fprintf(fp, "electron_data.esig_e[%d] = %15.5f\n", i,
                electron_data.esig_e[i]);
        fprintf(fp, "electron_data.psig_e[%d] = %15.5f\n", i,
                electron_data.psig_e[i]);
        fprintf(fp, "electron_data.expeke1[%d] = %15.5f\n", i,
                electron_data.expeke1[i]);
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.esig = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp, "esig0[%d][%d] = %15.5f, esig1[%d][%d] = %15.5f\n",
                    j, i, electron_data.esig0[idx],
                    j, i, electron_data.esig1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.psig = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp, "psig0[%d][%d] = %15.5f, psig1[%d][%d] = %15.5f\n",
                    j, i, electron_data.psig0[idx],
                    j, i, electron_data.psig1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.ededx = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp, "ededx0[%d][%d] = %15.5f, ededx1[%d][%d] = %15.5f\n",
                    j, i, electron_data.ededx0[idx],
                    j, i, electron_data.ededx1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.pdedx = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp, "pdedx0[%d][%d] = %15.5f, pdedx1[%d][%d] = %15.5f\n",
                    j, i, electron_data.pdedx0[idx],
                    j, i, electron_data.pdedx1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.ebr1 = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp, "ebr10[%d][%d] = %15.5f, ebr11[%d][%d] = %15.5f\n",
                    j, i, electron_data.ebr10[idx],
                    j, i, electron_data.ebr11[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.pbr1 = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp, "pbr10[%d][%d] = %15.5f, pbr11[%d][%d] = %15.5f\n",
                    j, i, electron_data.pbr10[idx],
                    j, i, electron_data.pbr11[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.pbr2 = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp, "pbr20[%d][%d] = %15.5f, pbr21[%d][%d] = %15.5f\n",
                    j, i, electron_data.pbr20[idx],
                    j, i, electron_data.pbr21[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.tmxs = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp, "tmxs0[%d][%d] = %15.5f, tmxs1[%d][%d] = %15.5f\n",
                    j, i, electron_data.tmxs0[idx],
                    j, i, electron_data.tmxs1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.e_array = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp, "e_array[%d][%d] = %15.5f\n",
                    j, i, electron_data.e_array[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.range_ep = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp, "range_ep[0][%d][%d] = %15.5f, range_ep[1][%d][%d] = %15.5f\n",
                    j, i, electron_data.range_ep[idx],
                    j, i, electron_data.range_ep[idx+geometry.nmed*MXEKE]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.etae_ms = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp,"etae_ms0[%d][%d] = %15.5f, etae_ms1[%d][%d] = %15.5f\n",
                    j, i, electron_data.etae_ms0[idx],
                    j, i, electron_data.etae_ms1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.etap_ms = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp,"etap_ms0[%d][%d] = %15.5f, etap_ms1[%d][%d] = %15.5f\n",
                    j, i, electron_data.etap_ms0[idx],
                    j, i, electron_data.etap_ms1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.q1ce_ms = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp,"q1ce_ms0[%d][%d] = %15.5f, q1ce_ms1[%d][%d] = %15.5f\n",
                    j, i, electron_data.q1ce_ms0[idx],
                    j, i, electron_data.q1ce_ms1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.q1cp_ms = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp,"q1cp_ms0[%d][%d] = %15.5f, q1cp_ms1[%d][%d] = %15.5f\n",
                    j, i, electron_data.q1cp_ms0[idx],
                    j, i, electron_data.q1cp_ms1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.q2ce_ms = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp,"q2ce_ms0[%d][%d] = %15.5f, q2ce_ms1[%d][%d] = %15.5f\n",
                    j, i, electron_data.q2ce_ms0[idx],
                    j, i, electron_data.q2ce_ms1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.q2cp_ms = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp,"q2cp_ms0[%d][%d] = %15.5f, q2cp_ms1[%d][%d] = %15.5f\n",
                    j, i, electron_data.q2cp_ms0[idx],
                    j, i, electron_data.q2cp_ms1[idx]);
        }
        fprintf(fp, "\n");
        
        fprintf(fp, "electron_data.blcce = \n");
        for (int j=0; j<MXEKE; j++) {
            int idx = i*MXEKE + j;
            fprintf(fp,"blcce0[%d][%d] = %15.5f, blcce1[%d][%d] = %15.5f\n",
                    j, i, electron_data.blcce0[idx],
                    j, i, electron_data.blcce1[idx]);
        }
        fprintf(fp, "\n");
        
    }
    
    fclose(fp);
    
    return;
}

void readRutherfordMscat(int nmed) {
    
    /* Get file path from input data */
    char data_folder[128];
    char buffer[128];
    
    if (getInputValue(buffer, "data folder") != 1) {
        printf("Can not find 'data folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(data_folder, buffer);
    
    char msnew_file[256];
    strcpy(msnew_file, data_folder);
    strcat(msnew_file, "msnew.data");
    
    /* Open multi-scattering file */
    FILE *fp;
    if ((fp = fopen(msnew_file, "r")) == NULL) {
        printf("Unable to open file: %s\n", msnew_file);
        exit(EXIT_FAILURE);
    }
    
    printf("Path to multi-scattering data file : %s\n", msnew_file);
    
    /* Allocate memory for MS data */
    mscat_data.ums_array =
        malloc((MXL_MS + 1)*(MXQ_MS + 1)*(MXU_MS + 1)*sizeof(double));
    mscat_data.fms_array =
        malloc((MXL_MS + 1)*(MXQ_MS + 1)*(MXU_MS + 1)*sizeof(double));
    mscat_data.wms_array =
        malloc((MXL_MS + 1)*(MXQ_MS + 1)*(MXU_MS + 1)*sizeof(double));
    mscat_data.ims_array =
        malloc((MXL_MS + 1)*(MXQ_MS + 1)*(MXU_MS + 1)*sizeof(double));
    
    printf("Reading multi-scattering data from file : %s\n", msnew_file);
    
    for (int i=0; i<=MXL_MS; i++) {
        for (int j=0; j <= MXQ_MS; j++) {
            int k, idx;
            
            for (k=0; k<=MXU_MS; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fscanf(fp, "%lf ", &mscat_data.ums_array[idx]);
            }
            for (k = 0; k<=MXU_MS; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fscanf(fp, "%lf ", &mscat_data.fms_array[idx]);
            }
            for (k = 0; k<=MXU_MS-1; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fscanf(fp, "%lf ", &mscat_data.wms_array[idx]);
            }
            for (k = 0; k<=MXU_MS-1; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fscanf(fp, "%d ", &mscat_data.ims_array[idx]);
            }
            
            for (k=0; k<=MXU_MS-1; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                mscat_data.fms_array[idx] = mscat_data.fms_array[idx + 1]/mscat_data.fms_array[idx] - 1.0;
                mscat_data.ims_array[idx] = mscat_data.ims_array[idx] - 1;
            }
            idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + MXU_MS;
            mscat_data.fms_array[idx] = mscat_data.fms_array[idx-1];
        }
    }

    double llammin = log(LAMBMIN_MS);
    double llammax = log(LAMBMAX_MS);
    double dllamb  = (llammax-llammin)/MXL_MS;
    mscat_data.dllambi = 1.0/dllamb;
    
    double dqms    = QMAX_MS/MXQ_MS;
    mscat_data.dqmsi = 1.0/dqms;
    
    fclose(fp);
    
    return;
}

void cleanMscat() {
    
    free(mscat_data.fms_array);
    free(mscat_data.ims_array);
    free(mscat_data.ums_array);
    free(mscat_data.wms_array);
    
    return;
}

void listMscat() {
    
    /* Get file path from input data */
    char output_folder[128];
    char buffer[128];
    
    if (getInputValue(buffer, "output folder") != 1) {
        printf("Can not find 'output folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(output_folder, buffer);
    
    char file_name[256];
    strcpy(file_name, output_folder);
    strcat(file_name, "mscat_data.lst");
    
    /* List mscat data to output file */
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
    fprintf(fp, "Listing multi-scattering data: \n");
    fprintf(fp, "dllambi = %f\n", mscat_data.dllambi);
    fprintf(fp, "dqmsi = %f\n", mscat_data.dqmsi);
    
    int idx;
    
    fprintf(fp, "\n");
    fprintf(fp, "ums_array = \n");
    for (int i=0; i<=MXL_MS; i++) {
        for (int j=0; j<=MXQ_MS; j++) {
            for (int k=0; k<=MXU_MS; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fprintf(fp, "ums_array[%d][%d][%d] = %f\n",
                        i, j, k, mscat_data.ums_array[idx]);
            }
        }
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "fms_array = \n");
    for (int i=0; i<=MXL_MS; i++) {
        for (int j=0; j<=MXQ_MS; j++) {
            for (int k=0; k<=MXU_MS; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fprintf(fp, "fms_array[%d][%d][%d] = %f\n",
                        i, j, k, mscat_data.fms_array[idx]);
            }
        }
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "wms_array = \n");
    for (int i=0; i<=MXL_MS; i++) {
        for (int j=0; j<=MXQ_MS; j++) {
            for (int k=0; k<=MXU_MS; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fprintf(fp, "wms_array[%d][%d][%d] = %f\n",
                        i, j, k, mscat_data.wms_array[idx]);
            }
        }
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "ims_array = \n");
    for (int i=0; i<=MXL_MS; i++) {
        for (int j=0; j<=MXQ_MS; j++) {
            for (int k=0; k<=MXU_MS; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fprintf(fp, "ims_array[%d][%d][%d] = %d\n",
                        i, j, k, mscat_data.ims_array[idx]);
            }
        }
    }
    fprintf(fp, "\n");
    
    fclose(fp);
    
    return;
}

void initSpinData(int nmed) {
    
    /* Get file path from input data */
    char data_folder[128];
    char buffer[128];
    
    if (getInputValue(buffer, "data folder") != 1) {
        printf("Can not find 'data folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(data_folder, buffer);
    
    char spinms_file[256];
    strcpy(spinms_file, data_folder);
    strcat(spinms_file, "spinms.data");
    
    /* Open spinms file */
    FILE *fp;
    if ((fp = fopen(spinms_file, "r")) == NULL) {
        printf("Unable to open file: %s\n", spinms_file);
        exit(EXIT_FAILURE);
    }
    
    printf("Path to spin data file : %s\n", spinms_file);
    
    /* Get length of file to create data buffers to reading */
    fseek(fp, 0, SEEK_END);
    long spin_file_len = ftell(fp);
    rewind(fp);
    
    /* The spin file is a binary one, therefore the reading process is much
     more strict. We use float variables as helpers */
    float *spin_buffer = (float*)malloc((spin_file_len/4)*sizeof(float));
    short *spin_buffer_int = (short*)malloc((spin_file_len/2)*sizeof(short));
    
    /* Read spin file version */
    char version[32];
    printf("\t");
    for (int i=0; i<32; i++) {
        fread(&version[i], 1, 1, fp);
        printf("%c", version[i]);
    }
    printf("\n");
    
    /* Read spin file endianess */
    char endianess[4];
    printf("\tspin file endianess : ");
    for (int i=0; i<4; i++) {
        fread(&endianess[i], 1, 1, fp);
        printf("%c", endianess[i]);
    }
    printf("\n");
    
    /* Read values for spin and b2, max and min values */
    float espin_max;
    float espin_min;
    float b2spin_max;
    float b2spin_min;
    fread(&espin_min, 4, 1, fp);
    fread(&espin_max, 4, 1, fp);
    fread(&b2spin_min, 4, 1, fp);
    fread(&b2spin_max, 4, 1, fp);
    
    /* Save information on spin data struct */
    spin_data.b2spin_min = (double)b2spin_min;
    
    float algo[276];
    fread(&algo, 263, 4, fp);
    
    int nener = MXE_SPIN;
    double dloge = log(espin_max/espin_min)/(double)nener;
    double eloge = log(espin_min);
    
    double *earray = (double*) malloc((MXE_SPIN1+1)*sizeof(double));
    earray[0] = espin_min;
    
    for (int i=1; i<=nener; i++) {
        eloge += dloge;
        earray[i] = exp(eloge);
    }
    
    double dbeta2 = (b2spin_max - b2spin_min)/nener;
    double beta2 = b2spin_min;
    earray[nener+1] = espin_max;
    
    for (int i=nener+2; i<=2*nener + 1; i++) {
        beta2 += dbeta2;
        
        if (beta2 < 0.999) {
            earray[i] = RM*1000.0*(1.0/sqrt(1.0 - beta2) - 1);
        }
        else {
            earray[i] = 50585.1;
        }
    }
    
    /* Convert to MeV and set interpolation intervals */
    espin_min /= 1000.0;
    espin_max /= 1000.0;
    double dlener = log(espin_max/espin_min)/MXE_SPIN;
    spin_data.dleneri = 1.0/dlener;
    spin_data.espml = log(espin_min);
    dbeta2 = (b2spin_max - b2spin_min)/MXE_SPIN;
    spin_data.dbeta2i = 1.0/dbeta2;
    double dqq1 = 0.5/MXQ_SPIN;
    spin_data.dqq1i = 1.0/dqq1;
    
    double *eta_array = (double*) malloc(2*(MXE_SPIN1+1)*sizeof(double));
    double *c_array = (double*) malloc(2*(MXE_SPIN1+1)*sizeof(double));
    double *g_array = (double*) malloc(2*(MXE_SPIN1+1)*sizeof(double));
    
    spin_data.spin_rej = malloc(nmed*2*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)*
                           (MXU_SPIN + 1)*sizeof(double));
    memset(spin_data.spin_rej, 0.0, nmed*2*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)*
           (MXU_SPIN + 1)*sizeof(double));
    
    double *fmax_array = (double*) malloc((MXQ_SPIN + 1)*sizeof(double));
    
    /* Needed for correction to first MS moment due to spin effects */
    double *elarray = (double*) malloc((MXE_SPIN1 + 1)*sizeof(double));
    double *farray = (double*) malloc((MXE_SPIN1 + 1)*sizeof(double));
    
    double *af = (double*) malloc((MXE_SPIN1+1)*sizeof(double));
    double *bf = (double*) malloc((MXE_SPIN1+1)*sizeof(double));
    double *cf = (double*) malloc((MXE_SPIN1+1)*sizeof(double));
    double *df = (double*) malloc((MXE_SPIN1+1)*sizeof(double));
    
    /* Allocate memory for electron data */
    electron_data.etae_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.etae_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.etap_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.etap_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.q1ce_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.q1ce_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.q1cp_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.q1cp_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.q2ce_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.q2ce_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.q2cp_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.q2cp_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.blcce0 = malloc(nmed*MXEKE*sizeof(double));
    electron_data.blcce1 = malloc(nmed*MXEKE*sizeof(double));
    
    /* Zero the following arrays, as they are surely not totally used. */
    memset(electron_data.etae_ms0, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.etae_ms1, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.etap_ms0, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.etap_ms1, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.q1ce_ms0, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.q1ce_ms1, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.q1cp_ms0, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.q1cp_ms1, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.q2ce_ms0, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.q2ce_ms1, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.q2cp_ms0, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.q2cp_ms1, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.blcce0, 0.0, nmed*MXEKE*sizeof(double));
    memset(electron_data.blcce1, 0.0, nmed*MXEKE*sizeof(double));
    
    for (int imed = 0; imed<nmed; imed++) {
        double sum_Z2 = 0.0, sum_A = 0.0, sum_pz = 0.0, sum_Z = 0.0;
        
        /* Set following arrays to zero before calculation */
        memset(eta_array, 0.0, 2*(MXE_SPIN1+1)*sizeof(double));
        memset(c_array, 0.0, 2*(MXE_SPIN1+1)*sizeof(double));
        memset(g_array, 0.0, 2*(MXE_SPIN1+1)*sizeof(double));
        
        rewind(fp);
        fread(&spin_buffer[0], 4, spin_file_len/4, fp);
        rewind(fp);
        fread(&spin_buffer_int[0], 2, spin_file_len/2, fp);
        
        int irec, i2_array[512], ii2;
        double dum1, dum2, dum3, aux_o, tau, eta, gamma, flmax;
        
        for (int i_ele=0; i_ele<pegs_data.ne[imed]; i_ele++) {
            double z = pegs_data.elements[imed][i_ele].z;
            int iz = (int)(z + 0.5);
            double pz = pegs_data.elements[imed][i_ele].pz;
            double tmp = z*(z + 1.0)*pz;
            
            sum_Z2 += tmp;
            sum_Z += pz*z;
            sum_A += pz*pegs_data.elements[imed][i_ele].wa;
            sum_pz += pz;
            double z23 = pow(z, 2.0/3.0);
            
            for (int iq = 0; iq<2; iq++) {
                for (int i=0; i<= MXE_SPIN1; i++) {
                    irec = 1 + (iz - 1)*4*(nener+1) + 2*iq*(nener + 1) + i + 1;
                    dum1 = spin_buffer[276*(irec - 1)];
                    dum2 = spin_buffer[276*(irec - 1) + 1];
                    dum3 = spin_buffer[276*(irec - 1) + 2];
                    aux_o = spin_buffer[276*(irec - 1) + 3];
                    
                    for (int fai = 0; fai <= MXQ_SPIN; fai++) {
                        fmax_array[fai] = spin_buffer[276 * (irec - 1)
                                                      + 4 + fai];
                    }
                    for (int i2 = 0; i2<512; i2++) {
                        i2_array[i2] = spin_buffer_int[552 * (irec - 1)
                                                       + 40 + i2];
                    }
                    
                    eta_array[iq*(MXE_SPIN1+1) + i] += tmp*log(z23*aux_o);
                    
                    /* Energy in the file is in keV */
                    tau = earray[i]/(1000.0*RM);
                    beta2 = tau*(tau + 2)/((tau + 1)*(tau + 1));
                    eta = z23/((137.03604/*fine*/*0.88534138/*TF_constant*/)*
                                 (137.03604*0.88534138))*aux_o/4/tau/(tau + 2);
                    c_array[iq*(MXE_SPIN1+1) + i] += tmp*(log(1.0 + 1.0/eta)
                        - 1.0 / (1.0 + eta))*dum1*dum3;
                    g_array[iq*(MXE_SPIN1+1) + i] += tmp*dum2;
                    
                    for (int j = 0; j <= MXQ_SPIN; j++) {
                        for (int k = 0; k <= MXU_SPIN; k++) {
                            ii2 = (int)i2_array[(MXU_SPIN + 1)*j + k];
                            if (ii2<0) { ii2 += 65536; }
                            dum1 = ii2;
                            dum1 = dum1*fmax_array[j] / 65535;
                            spin_data.spin_rej[imed*2*(MXE_SPIN1 + 1)*
                                     (MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                     + iq*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)
                                     *(MXU_SPIN + 1)
                                     + i*(MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                     + j*(MXU_SPIN + 1) + k] += tmp*dum1;
                        }
                    }
                }
            }
        }
        
        /* spin_rej will be used as a rejection function in MS sampling, so
         scale maximum to unity */
        for (int iq=0; iq<2; iq++) {
            for (int i=0; i<=MXE_SPIN1; i++) {
                for (int j=0; j<=MXQ_SPIN; j++) {
                    flmax = 0.0;
                    for (int k=0; k<=MXU_SPIN; k++) {
                        if (flmax<spin_data.spin_rej[imed*2*(MXE_SPIN1 + 1)*
                                           (MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                           + iq*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)
                                           *(MXU_SPIN + 1)
                                           + i*(MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                           + j*(MXU_SPIN + 1) + k]) {
                            flmax = spin_data.spin_rej[imed*2*(MXE_SPIN1 + 1)*
                                             (MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                             + iq*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)
                                             *(MXU_SPIN + 1)
                                             + i*(MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                             + j*(MXU_SPIN + 1) + k];
                        }
                    }
                    for (int k = 0; k <= MXU_SPIN; k++) {
                        spin_data.spin_rej[imed*2*(MXE_SPIN1 + 1)*
                                 (MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                 + iq*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)
                                 *(MXU_SPIN + 1)
                                 + i*(MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                 + j*(MXU_SPIN + 1) + k] =
                        spin_data.spin_rej[imed*2*(MXE_SPIN1 + 1)*
                                 (MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                 + iq*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)
                                 *(MXU_SPIN + 1)
                                 + i*(MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                 + j*(MXU_SPIN + 1) + k]/flmax;
                    }
                }
            }
        }
        
        /* Process eta_array, c_array and g_array to their final form */
        for (int i=0; i <= MXE_SPIN1; i++) {
            tau = (earray[i]/RM)*0.001;
            beta2 = tau*(tau + 2.0)/pow(tau + 1.0, 2.0);
            
            for (int iq=0; iq<2; iq++) {
                aux_o = exp(eta_array[iq*(MXE_SPIN1 + 1) + i]/sum_Z2) /
                (pow(137.03604*0.88534138, 2.0));
                eta_array[iq*(MXE_SPIN1 + 1) + i] = 0.26112447*aux_o*
                (electron_data.blcc[imed])/(electron_data.xcc[imed]);
                eta = aux_o / 4.0 / tau / (tau + 2);
                gamma = 3.0*(1.0 + eta)*(log(1.0 + 1.0 / eta)*(1.0 + 2.0*eta)
                                         - 2.0) /
                (log(1.0 + 1.0 / eta)*(1.0 + eta) - 1.0);
                g_array[iq*(MXE_SPIN1 + 1) + i] =
                    g_array[iq*(MXE_SPIN1 + 1) + i]/sum_Z2/gamma;
                c_array[iq*(MXE_SPIN1 + 1) + i] =
                    c_array[iq*(MXE_SPIN1 + 1) + i]/sum_Z2/
                    (log(1.0 + 1.0/eta) - 1.0/(1.0 + eta));
            }
        }
        
        /* Prepare interpolation table for the screening parameter */
        double eil = (1.0 - electron_data.eke0[imed])/electron_data.eke1[imed];
        double e = exp(eil);
        double si1e, si1p, si2e, si2p, aae;
        int je = 0;
        
        if (e<=espin_min) {
            si1e = eta_array[0*(MXE_SPIN1 + 1) + 0];
            si1p = eta_array[1*(MXE_SPIN1 + 1) + 0];
        }
        else {
            if (e <= espin_max) {
                aae = (eil - spin_data.espml)*spin_data.dleneri;
                je = (int)aae;
                aae = aae - je;
            }
            else {
                tau = e/RM;
                beta2 = tau*(tau + 2.0)/pow(tau + 1.0, 2.0);
                aae = (beta2 - spin_data.b2spin_min)*spin_data.dbeta2i;
                je = (int)aae;
                aae = aae - je;
                je = je + MXE_SPIN + 1;
            }
            si1e = (1 - aae)*eta_array[0*(MXE_SPIN1 + 1) + je] +
                aae*eta_array[0*(MXE_SPIN1 + 1) + (je + 1)];
            si1p = (1 - aae)*eta_array[1*(MXE_SPIN1 + 1) + je] +
                aae*eta_array[1*(MXE_SPIN1 + 1) + (je + 1)];
        }
        
        int neke = pegs_data.meke[imed];
        for (int i=1; i<neke; i++) {
            eil = (i + 1.0 - electron_data.eke0[imed])/electron_data.eke1[imed];
            e = exp(eil);
            if (e<=espin_min) {
                si2e = eta_array[0*(MXE_SPIN1 + 1) + 0];
                si2p = eta_array[1*(MXE_SPIN1 + 1) + 0];
            }
            else {
                if (e<=espin_max) {
                    aae = (eil - spin_data.espml)*spin_data.dleneri;
                    je = (int)aae;
                    aae = aae - je;
                }
                else {
                    tau = e/RM;
                    beta2 = tau*(tau + 2.0)/((tau + 1.0)*(tau + 1.0));
                    aae = (beta2 - spin_data.b2spin_min)*spin_data.dbeta2i;
                    je = (int)aae;
                    aae = aae - je;
                    je = je + MXE_SPIN + 1;
                }
                si2e = (1.0 - aae)*eta_array[0*(MXE_SPIN1 + 1) + je] +
                    aae*eta_array[0*(MXE_SPIN1 + 1) + (je + 1)];
                si2p = (1.0 - aae)*eta_array[1*(MXE_SPIN1 + 1) + je] +
                    aae*eta_array[1*(MXE_SPIN1 + 1) + (je + 1)];
                
            }
            
            electron_data.etae_ms1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron_data.eke1[imed];
            electron_data.etae_ms0[MXEKE*imed + i - 1] =
                (si2e - electron_data.etae_ms1[MXEKE*imed + i - 1]*eil);
            electron_data.etap_ms1[MXEKE*imed + i - 1] =
                (si2p - si1p)*electron_data.eke1[imed];
            electron_data.etap_ms0[MXEKE*imed + i - 1] =
                (si2p - electron_data.etap_ms1[MXEKE*imed + i - 1]*eil);
            si1e = si2e; si1p = si2p;
        }
        
        electron_data.etae_ms1[MXEKE*imed + neke - 1] =
            electron_data.etae_ms1[MXEKE*imed + neke - 2];
        electron_data.etae_ms0[MXEKE*imed + neke - 1] =
            electron_data.etae_ms0[MXEKE*imed + neke - 2];
        electron_data.etap_ms1[MXEKE*imed + neke - 1] =
            electron_data.etap_ms1[MXEKE*imed + neke - 2];
        electron_data.etap_ms0[MXEKE*imed + neke - 1] =
            electron_data.etap_ms0[MXEKE*imed + neke - 2];

        /* Prepare correction to the first MS moment due to spin effects */
        /* First electrons */
        for (int i=0; i<=MXE_SPIN; i++) {
            elarray[i] = log(earray[i]/1000.0);
            farray[i] = c_array[0*(MXE_SPIN1 + 1) + i];
        }
        for (int i=MXE_SPIN+1; i<=MXE_SPIN1 - 1; i++) {
            elarray[i] = log(earray[i+1]/1000.0);
            farray[i] = c_array[0*(MXE_SPIN1 + 1) + i + 1];
        }
        
        int ndata = MXE_SPIN1 + 1;
        if (pegs_data.ue[imed] > 1.0E5) {
            elarray[ndata - 1] = log(pegs_data.ue[imed]);
        }
        else {
            elarray[ndata - 1] = log(1.0E5);
        }
        farray[ndata - 1] = 1.0;
        
        setSpline(elarray, farray, af, bf, cf, df, ndata);
        eil = (1.0 - electron_data.eke0[imed])/electron_data.eke1[imed];
        si1e = spline(eil, elarray, af, bf, cf, df, ndata);
        
        for(int i=1; i<=neke - 1; i++){
            eil = (i + 1 - electron_data.eke0[imed])/electron_data.eke1[imed];
            si2e = spline(eil, elarray, af, bf, cf, df, ndata);
            electron_data.q1ce_ms1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron_data.eke1[imed];
            electron_data.q1ce_ms0[MXEKE*imed + i - 1]=
                si2e - electron_data.q1ce_ms1[MXEKE*imed + i - 1]*eil;
            si1e = si2e;
        }
        electron_data.q1ce_ms1[MXEKE*imed + neke - 1] =
            electron_data.q1ce_ms1[MXEKE*imed + neke - 2];
        electron_data.q1ce_ms0[MXEKE*imed + neke - 1] =
            electron_data.q1ce_ms1[MXEKE*imed + neke - 2];
        
        /* Now positrons */
        for (int i=0; i<=MXE_SPIN; i++){
            farray[i] = c_array[1*(MXE_SPIN1 + 1) + i];
        }
        for (int i=MXE_SPIN+1; i<=MXE_SPIN1 - 1; i++){
            farray[i] = c_array[1*(MXE_SPIN1 + 1) + i + 1];
        }
        
        setSpline(elarray, farray, af, bf, cf, df, ndata);
        eil = (1.0 - electron_data.eke0[imed])/electron_data.eke1[imed];
        si1e = spline(eil, elarray, af, bf, cf, df, ndata);
        
        for (int i=1; i<=neke-1; i++){
            eil = (i + 1 - electron_data.eke0[imed])/electron_data.eke1[imed];
            si2e = spline(eil, elarray, af, bf, cf, df, ndata);
            electron_data.q1cp_ms1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron_data.eke1[imed];
            electron_data.q1cp_ms0[MXEKE*imed + i - 1]=
                si2e - electron_data.q1cp_ms1[MXEKE*imed + i - 1]*eil;
            si1e = si2e;
        }
        electron_data.q1cp_ms1[MXEKE*imed + neke - 1] =
            electron_data.q1cp_ms1[MXEKE*imed + neke - 2];
        electron_data.q1cp_ms0[MXEKE*imed + neke - 1] =
            electron_data.q1cp_ms0[MXEKE*imed + neke - 2];
        
        /* Prepare interpolation table for the second MS moment correction */
        /* First electrons */
        for (int i=0; i<=MXE_SPIN; i++) {
            farray[i] = g_array[0*(MXE_SPIN1+1) + i];
        }
        for (int i=MXE_SPIN + 1; i<=MXE_SPIN1 - 1; i++) {
            farray[i] = g_array[0*(MXE_SPIN1+1) + i + 1];
        }
        
        setSpline(elarray, farray, af, bf, cf, df, ndata);
        eil = (1.0 - electron_data.eke0[imed])/electron_data.eke1[imed];
        si1e = spline(eil, elarray, af, bf, cf, df, ndata);
        
        for (int i=1; i<=neke-1; i++){
            eil = (i + 1 - electron_data.eke0[imed])/electron_data.eke1[imed];
            si2e = spline(eil, elarray, af, bf, cf, df, ndata);
            electron_data.q2ce_ms1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron_data.eke1[imed];
            electron_data.q2ce_ms0[MXEKE*imed + i - 1] =
                si2e - electron_data.q2ce_ms1[MXEKE*imed + i - 1]*eil;
            si1e = si2e;
        }
        electron_data.q2ce_ms1[MXEKE*imed + neke - 1] =
            electron_data.q2ce_ms1[MXEKE*imed + neke - 2];
        electron_data.q2ce_ms0[MXEKE*imed + neke - 1] =
            electron_data.q2ce_ms0[MXEKE*imed + neke - 2];
        
        /* Now positrons */
        for (int i=0; i<=MXE_SPIN; i++){
            farray[i] = g_array[1*(MXE_SPIN1 + 1) + i];
        }
        for (int i=MXE_SPIN + 1; i<=MXE_SPIN1 - 1; i++){
            farray[i] = g_array[1*(MXE_SPIN1 + 1) + i + 1];
        }
        
        setSpline(elarray, farray, af, bf, cf, df, ndata);
        eil = (1.0 - electron_data.eke0[imed])/electron_data.eke1[imed];
        si1e = spline(eil, elarray, af, bf, cf, df, ndata);
        
        for (int i=1; i<=neke - 1; i++){
            eil = (i + 1 - electron_data.eke0[imed])/electron_data.eke1[imed];
            si2e = spline(eil, elarray, af, bf, cf, df, ndata);
            electron_data.q2cp_ms1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron_data.eke1[imed];
            electron_data.q2cp_ms0[MXEKE*imed + i - 1] =
                si2e - electron_data.q2cp_ms1[MXEKE*imed + i - 1]*eil;
            si1e = si2e;
        }
        electron_data.q2cp_ms1[MXEKE*imed + neke - 1] =
            electron_data.q2cp_ms1[MXEKE*imed + neke - 2];
        electron_data.q2cp_ms0[MXEKE*imed + neke - 1] =
            electron_data.q2cp_ms0[MXEKE*imed + neke - 2];
        
        /* Now substract scattering power that is already taken into account in
         discrete Moller/Bhabha events */
        double tauc = pegs_data.te[imed]/RM;
        int leil;
        double dedx, etap, g_r, g_m, sig;
        si1e=1.0;
        
        for (int i=1; i<=neke - 1; i++){
            eil = ((double)(i + 1) - electron_data.eke0[imed])/electron_data.eke1[imed];
            e = exp(eil);
            leil = i;
            tau = e/RM;
            
            if (tau > 2.0*tauc){
                sig = electron_data.esig1[MXEKE*imed + leil]*eil +
                    electron_data.esig0[MXEKE*imed + leil];
                dedx = electron_data.ededx1[MXEKE*imed + leil]*eil +
                    electron_data.ededx0[MXEKE*imed + leil];
                sig /= dedx;
                
                if (sig>1.0E-6) { /* to be sure that this is not a CSDA calc. */
                    etap = electron_data.etae_ms1[MXEKE*imed + leil]*eil +
                        electron_data.etae_ms0[MXEKE*imed + leil];
                    eta = 0.25*etap*(electron_data.xcc[imed])/
                        (electron_data.blcc[imed])/tau/(tau+2);
                    g_r = (1.0 + 2.0*eta)*log(1.0 + 1.0/eta) - 2.0;
                    g_m = log(0.5*tau/tauc) + (1.0 + ((tau + 2.0)/(tau + 1.0))*
                        ((tau + 2.0)/(tau + 1.0)))*log(2.0*(tau - tauc + 2.0)/
                        (tau + 4.0)) - 0.25*(tau + 2.0)*
                        (tau + 2.0 + 2.0*(2.0*tau + 1.0)/
                        ((tau + 1.0)*(tau + 1.0)))*log((tau + 4.0)*(tau - tauc)
                        /tau/(tau - tauc + 2.0)) +
                        0.5*(tau - 2.0*tauc)*(tau + 2.0)*(1.0/(tau - tauc) -
                        1.0/((tau + 1.0)*(tau + 1.0)));
                    
                    if (g_m < g_r){
                        g_m /= g_r;
                    }
                    else{
                        g_m = 1.0;
                    }
                    si2e = 1.0 - g_m*sum_Z/sum_Z2;
                }
                else{
                    si2e = 1.0;
                }
            }
            else{
                si2e = 1.0;
            }
            
            electron_data.blcce1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron_data.eke1[imed];
            electron_data.blcce0[MXEKE*imed + i - 1] =
                si2e - electron_data.blcce1[MXEKE*imed + i - 1]*eil;
            si1e = si2e;
        }
        electron_data.blcce1[MXEKE*imed + neke - 1] =
            electron_data.blcce1[MXEKE*imed + neke - 2];
        electron_data.blcce0[MXEKE*imed + neke - 1] =
            electron_data.blcce0[MXEKE*imed + neke - 2];
        
    }
    
    fclose(fp);
    
    /* Cleaning */
    free(earray);
    free(eta_array);
    free(c_array);
    free(g_array);
    free(elarray);
    free(farray);
    free(af);
    free(bf);
    free(cf);
    free(df);
    
    return;
}

void cleanSpin() {
    
    free(spin_data.spin_rej);
    
    return;
}

void listSpin() {
    
    /* Get file path from input data */
    char output_folder[128];
    char buffer[128];
    
    if (getInputValue(buffer, "output folder") != 1) {
        printf("Can not find 'output folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(output_folder, buffer);
    
    char file_name[256];
    strcpy(file_name, output_folder);
    strcat(file_name, "spin_data.lst");
    
    /* List spin data to output file */
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
    fprintf(fp, "Listing spin data: \n");
    fprintf(fp, "b2spin_min = %f\n", spin_data.b2spin_min);
    fprintf(fp, "dbeta2i = %f\n", spin_data.dbeta2i);
    fprintf(fp, "espml = %f\n", spin_data.espml);
    fprintf(fp, "dleneri = %f\n", spin_data.dleneri);
    fprintf(fp, "dqq1i = %f\n", spin_data.dqq1i);
    fprintf(fp, "\n");
    
    int idx;
    
    for (int imed=0; imed<geometry.nmed; imed++) {
        fprintf(fp, "For medium %s: \n", geometry.med_names[imed]);
        fprintf(fp, "spin_rej = \n");
        
        for (int i=0; i<=MXE_SPIN1; i++) {
            for (int j=0; j<= MXQ_SPIN; j++) {
                for (int k=0; k<=MXU_SPIN; k++) {
                    idx = imed*2*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)*(MXU_SPIN + 1)
                    + i*(MXQ_SPIN + 1)*(MXU_SPIN + 1)
                    + j*(MXU_SPIN + 1) + k;
                    fprintf(fp, "spin_rej[%d][0][%d][%d][%d] = %15.5f, "
                            "spin_rej[%d][1][%d][%d][%d] = %15.5f\n",
                            imed, i, j, k, spin_data.spin_rej[idx],
                            imed, j, i, k, spin_data.spin_rej[idx +
                                (MXE_SPIN1 + 1)*(MXQ_SPIN + 1)*(MXU_SPIN + 1)]);
                    
                }
            }
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    
    return;
}

void setSpline(double *x, double *f, double *a, double *b, double *c,
                double *d,int n) {
    
    double s,r;
    int m1,m2,m,mr;
    m1=2;
    m2=n-1;
    s=0;
    for (m=1;m<=m2;m++){
        d[m-1]=x[m]-x[m-1];
        r=(f[m]-f[m-1])/d[m-1];
        c[m-1]=r-s;
        s=r;
    }
    s=0;
    r=0;
    c[0]=0;
    c[n-1]=0;
    
    for(m=m1; m<=m2; m++){
        c[m-1]=c[m-1]+r*c[m-2];
        b[m-1]=2*(x[m-2]-x[m])-r*s;
        s=d[m-1];
        r=s/b[m-1];
    }
    mr=m2;
    
    for(m=m1; m<=m2; m++){
        c[mr-1]=(d[mr-1]*c[mr]-c[mr-1])/b[mr-1];
        mr=mr-1;
    }
    
    for(int m=1; m<=m2; m++){
        s=d[m-1];
        r=c[m]-c[m-1];
        d[m-1]=r/s;
        c[m-1]=3*c[m-1];
        b[m-1]=(f[m]-f[m-1])/s-(c[m-1]+r)*s;
        a[m-1]=f[m-1];
    }
    
    return;
}

double spline(double s, double *x, double *a, double *b, double *c,
              double *d, int n) {
    
    int  m_lower,m_upper,direction,m,ml,mu,mav;
    double q;
    if( x[0] > x[n-1] ) {
        direction = 1;
        m_lower = n;
        m_upper = 0;
    }
    else {
        direction = 0;
        m_lower = 0;
        m_upper = n;
    }
    if ( s >= x[m_upper + direction-1] ) {
        m = m_upper + 2*direction - 1;
    }
    else if( s <= x[m_lower-direction]) {
        m = m_lower - 2*direction + 1;
    }
    else {
        /* Perform a binary search to find the interval s is in */
        ml = m_lower; mu = m_upper;
        while ( abs(mu-ml) > 1 ) {
            mav = (ml+mu)/2;
            if( s < x[mav-1] ) { mu = mav; }
            else             { ml = mav; }
        }
        m = mu + direction - 1;
    }
    q = s - x[m-1];
    double spline = a[m-1] + q*(b[m-1]+ q*(c[m-1] + q*d[m-1]));
    return spline;
    
}

void electron() {
    
    int np = stack.np;              // stack pointer
    int irl = stack.ir[np];         // region index
    int imed = region.med[irl];     // medium index of current region
    double rhof = region.rhof[irl]; // mass density ratio
    double edep = 0.0;              // deposited energy by particle
    
    struct Uphi uphi;
    double eie = stack.e[np];       // energy of incident electron
    int iq = stack.iq[np];         // charge of current particle.
    int qel = (1 + iq)/2;           // = 0 for electrons, = 1 for positrons
    int medold; 

    double rnno;

    /* First check of electron cut-off energy */
    if(eie <= region.ecut[irl]) {
        
        edep = stack.e[np] - RM;    // get energy deposition for user

        /* Call ausgab and drop energy on spot */
        ausgab(edep);
        
        /* Positron annihilation section */
        if(iq > 0) {
            /* The particle is a positron. Produce annihilation gammas
             if edep < eie */
            if(edep < eie) {
                rannih();
                
                // Now discard the positron and take normal return to
                // follow the annihilation gammas.
                return;
            }
        }
        
        stack.np -= 1;
        return;
    }
    
    double elke;    // logarithm of kinetic energy
    int lelke;   // index into the energy grid of tabulated funtions
    double sigratio = 0.0;
    double rfict = 0.0; // rejection function for fictitius cross section.

    do {    // start of tstep loop
        
        /* Go through this loop each time we recompute distance to an
         interaction */
        int compute_tstep = 1;  // mean free path resampled. Calculate
                                // distance to the interaction in ustep loop
        
        double eke = eie - RM;  // kinetic energy of the particle
        double demfp;           // differential electron mean free path
        
        double sigf;            // cross section before density scaling but
                                // after a step
        double sig0;            // cross section before density scaling but
                                // before a step
        double dedx0;           // stopping power before density scaling
        
        double ustep;            // projected transport distance in the
                                // direction of motion at the start of the step
        
        if(imed != -1) {
            /* Not vacuum. The electron mean free path must be sampled to 
            determine how far is the next interaction */
            
            /* Select electron mean free path */
            rnno = setRandom();
            if(rnno == 0.0) {
                rnno = 1.0E-30;
            }
            demfp = fmax(-log(rnno), EPSEMFP);
            
            /* Prepare to aproximate cross section. First obtain energy
                interval of current particle */
            elke = log(eke);
            
            /* lelke adjusted to C standard */
            lelke = pwlfInterval(imed, elke, electron_data.eke1,
                                    electron_data.eke0) - 1;

            /* Evaluate sig0 for the fictitious method. This version uses
            sub-threshold energy loss as a measure of path-length. Cross
            section is actual cross section divided by restricted
            stopping power */

            if(electron_data.sig_ismonotone[qel*geometry.nmed+imed]) {
                if(iq < 0) {
                    sig0 = pwlfEval(imed*MXEKE+lelke, elke, 
                        electron_data.esig1, electron_data.esig0);
                    dedx0 = pwlfEval(imed*MXEKE+lelke, elke, 
                        electron_data.ededx1, electron_data.ededx0);
                    sig0 /= dedx0;
                }
                else {
                    sig0 = pwlfEval(imed*MXEKE+lelke, elke, 
                        electron_data.psig1, electron_data.psig0);
                    dedx0 = pwlfEval(imed*MXEKE+lelke, elke, 
                        electron_data.pdedx1, electron_data.pdedx0);
                    sig0 /= dedx0;
                }
            }
            else {
                /* Use the global maximum values determined in the host */
                if(iq < 0) {
                    sig0 = electron_data.esig_e[imed];
                }
                else {
                    sig0 = electron_data.psig_e[imed];
                }
            }
            
        } // end of non-vacuum test
        
        do {    // start of ustep loop
            /* For each particle check step distance with user geometry. 
            Compute size of maximum acceptable step, which is limited by 
            multiple scattering or other approximations */

			int call_howfar;    // indicates if the boundary crossing 
								// algorithm (BCA) requires a call 
								// to howfar
			
			int do_single;      // if true, exact BCA requires single scattering
			int called_msdist;  // true, normal CH transport, false, BCA invoked

			double tstep;   // total pathlength to the next discrete interaction
			double tustep;	// total pathlength of the electron step

			double total_de;    // total energy loss to next discrete interaction
			double ekef;		// kinetic energy after a step

			double ekei;        // used to calculate tstep from demfp
			double elkei;	    // log(ekei)
			double tuss;	    // sampled path-length to a single scattering 
                                // event
			double total_tstep; // total path-length to next discrete 
                                // interaction

			double range;       // electron range

			double p2;      // electron momentum times c, squared
			double beta2;   // electron speed in units of c, squared
			double etap;	// Correction to Moliere screening angle from 
                            // PWA cross sections

			double tvstep;	// curved path-length calculated from tustep
			double de;		// energy loss to dedx

			double x_final; // position at the end of step
            double y_final; 
            double z_final; 
			double u_final; // direction at the end of step
            double v_final;
            double w_final;

			if (imed == -1) {   // vacuum
				tstep = 10.0E8; // i.e. infinity
				ustep = tstep;
				tustep = ustep;
				call_howfar = 1;	// always howfar is called for vacuum 
									// steps
			}
			else {  // non-vacuum
				/* Update density of medium */
                rhof = region.rhof[irl];

				/* As the cross-section is interactions per energy loss, no 
                density scaling is required here */
				if (sig0 <= 0.0) {
					/* This can happen if the threshold for brems (ap + rm)
					is greater than ae. Ask for step same as vacuum */
					tstep = 10.0E8; //  i.e. infinity
					sig0 = 1.0E-15;
				}
				else {
					/* Calculate tstep from differential electron mean 
					free path. Once the sub-threshold processes energy
					loss to the next discrete interaction is determined,
					the corresponding path-length is calculated */

					if (compute_tstep) {
						total_de = demfp/sig0;
						ekef = eke - total_de;

						if (ekef <= electron_data.e_array[imed*MXEKE+0]) {
							tstep = 10.0E8; // i.e. infinity
						}
						else {
							double elkef = log(ekef);

							/* lelkef adjusted to C standard */
							int lelkef = pwlfInterval(imed, elkef, 
                                electron_data.eke1, electron_data.eke0) - 1;

							if (lelkef == lelke) {
								/* Initial and final energy are in the same
								interval of the PWLF function */

								/* The following computes the path-length 
								traveled while going from energy eke to ekef */
								tstep = computeDrange(imed, iq, lelke, 
                                    eke, ekef, elke, elkef);

							}
							else {
								/* Initial and final energy are in 
								different intervals of the PWL function */

								/* The calculation of the path length must be 
                                divided among the intervals of the PWL 
                                function */

								/* Calculate range from ekef to E(lelkfef+1) 
                                and from E(lelke) to eke and add the pre-calc 
								range from E(lelfke+1) to E(lelke) */

								/* First calculate range from eke to E(lelke) */
								ekei = electron_data.e_array[imed*MXEKE+lelke];
								elkei = (lelke+1 - electron_data.eke0[imed])/
                                    electron_data.eke1[imed];
								tuss = computeDrange(imed, iq, lelke, 
                                    eke, ekei, elke, elkei);

								/* Then from E(lelfke+1) to ekef */
								ekei = electron_data.e_array[imed*MXEKE+
                                    lelkef+1];
								elkei = ((lelkef+2) - electron_data.eke0[imed])/
                                    electron_data.eke1[imed];
								tstep = computeDrange(imed, iq, lelkef, 
                                    ekei, ekef, elkei, elkef);

								/* Finally add the pre-calc range from 
                                E(lelfke+1) to E(lelke) */
								tstep += tuss + 
            electron_data.range_ep[qel*geometry.nmed*MXEKE+imed*MXEKE+lelke] - 
            electron_data.range_ep[qel*geometry.nmed*MXEKE+imed*MXEKE+lelkef+1];
							}
						}

						total_tstep = tstep;
						compute_tstep = 0;  // i.e. false
					}   // end of compute_tstep if sentence

					tstep = total_tstep/rhof; // non-default density scaling

				}	// end sig if-else

				/* Calculate stopping power */
				if (iq < 0) {   // electron
					dedx0 = pwlfEval(imed*MXEKE+lelke, elke, 
                        electron_data.ededx1, electron_data.ededx0);
				}
				else {  // positron
					dedx0 = pwlfEval(imed*MXEKE+lelke, elke, 
                        electron_data.pdedx1, electron_data.pdedx0);
				}
				double dedx = rhof*dedx0;   // stopping power after density 
                                            // scaling

				/* Determine maximum step-size */
				double tmxs = pwlfEval(imed*MXEKE+lelke, elke, 
                    electron_data.tmxs1, electron_data.tmxs0);
				tmxs /= rhof;

				/* Compute the range to E_min(med), where e_min is the 
				first energy in the table. Limit the electron step to 
				this range */
                ekei = electron_data.e_array[imed*MXEKE+lelke];
                elkei = (lelke+1 - electron_data.eke0[imed])/
                        electron_data.eke1[imed];
                range = computeDrange(imed, iq, lelke, eke, ekei, 
                        elke, elkei);
                range += electron_data.range_ep[qel*geometry.nmed*MXEKE+
                    imed*MXEKE+lelke];
				range /= rhof;

				/* The following finds the minimum between tstep, tmxs and 
				range. The result is stored in tustep, the total 
				path-length to the next interaction */
				tustep = fmin(fmin(tstep, tmxs), range);

				/* Obtain perpendicular distance to nearest boundary */
				double tperp = hownear();
				stack.dnear[np] = tperp;

				/* Set the minimum step size for a CH step, due to efficiency 
				considerations. It is calculated with eke and elke */
				double blccl = rhof*electron_data.blcc[imed];
				double xccl = rhof*electron_data.xcc[imed];
				p2 = eke*(eke+2.0*RM);
				beta2 = p2/(p2 + pow(RM, 2.0));

				/* Now calculate the elastic scattering MFP, based on PWA
				cross sections */
				if (iq < 0) {
					etap = pwlfEval(MXEKE*imed+lelke, elke, 
                        electron_data.etae_ms1, electron_data.etae_ms0);
				}
				else {
					etap = pwlfEval(MXEKE*imed+lelke, elke, 
                        electron_data.etap_ms1, electron_data.etap_ms0);
				}

				double ms_corr = pwlfEval(MXEKE*imed+lelke, elke, 
                    electron_data.blcce1, electron_data.blcce0);
				blccl = blccl/etap/(1.0 + 0.25*etap*xccl/blccl/p2)*ms_corr;

				double ssmfp = beta2/blccl;   // mean free path to one single 
                                                // elastic scattering event

				/* Finally, set the the minimum CH step size */
				double skindepth = SKIN_DEPTH_FOR_BCA*ssmfp;

				/* Adjust tustep with respect to perpendicular distance to
				closest boundary and minimum CH step size */
				tustep = fmin(tustep, fmax(tperp, skindepth));

				/* The transport logic below is determined by the boolean 
				variables call_howfar and do_single */
				int is_chstep = 0;  // i.e. false
                
				if ((tustep <= tperp) && (tustep > skindepth)) {
					/* The particle is further a boundary than skindepth, 
					so perform a normal CH step */

					call_howfar = 0;    // do not call howfar
					do_single = 0;      // ms => no single scattering
					called_msdist = 1;  // remember than msdist has been called

					/* Compute energy loss due to sub-threshold processes 
					for a path-length tustep */
					de = computeEloss(imed, iq, irl, rhof,
					    tustep,	range, eke, elke, lelke);

					tvstep = tustep;
					is_chstep = 1;  // i.e. true

					/* msdist models multiple elastic scattering and 
					spatial deflections for the path-length tustep.
					ustep is the straight-line distance between 
					initial and final position of the particle */                    
					ustep = msdist(imed, iq, rhof, de, tustep, eke,
                    	&x_final, &y_final, &z_final,
                        &u_final, &v_final, &w_final);                    
				}
				else {
                    
					/* We are within a skindepth from a boundary, invoke
					the boundary-crossing algorithm (exact) */					
					called_msdist = 0;  // remember that msdist has not 
										// been called.

					/* Now cross the boundary in single scattering mode.
					We use always exact BCA */

					/* sample the distance to a single scattering event */
					rnno = setRandom();
					if (rnno < 1.0E-30) {
						rnno = 1.0E-30;
					}

                    /* Calculate number of mean free paths (elastic scattering cross-section)*/
					double lambda = (-1.0)*log(1.0 - rnno); 
					double lambda_max = 0.5*blccl*RM/dedx*pow((eke/RM+1.0),3.0);
                                        
                    if (lambda >= 0.0 && lambda_max > 0.0) {
                        if (lambda < lambda_max) {
						tuss = lambda*ssmfp*(1.0 - 0.5*lambda/lambda_max);
                        }
                        else {
                            tuss = 0.5*lambda*ssmfp;
                        }
                        if (tuss < tustep) {
                            tustep = tuss;
                            do_single = 1;  // i.e. true
                        }
                        else {
                            do_single = 0;  // i.e. false
                        }    
                    }
                    else {
                        printf("Warning!, lambda = %f > lambda_max = %f\n", 
                            lambda, lambda_max);
                        do_single = 0;
                        stack.np -= 1;
                        return;
                    }
                    
					ustep = tustep;
					if (ustep < tperp) {
						call_howfar = 0;
					}
					else {
						call_howfar = 1;
					}
				} // end of skindepth if-else
			}   // end of non-vacuum if-else 
			
			int irold = stack.ir[np];   // region before transport
            int irnew = stack.ir[np];   // default new region is old region
			int idisc = 0;		        // default is no discard
			
			if (call_howfar) {
				howfar(&idisc, &irnew, &ustep);
			}			
			
			/* Now see if user requested discard through idisc, returned 
			by howfar */
			if (idisc > 0) {
				/* User requested electron discard */
                if(iq > 0) {
                    edep = stack.e[np] + RM;
                }
                else {
                    edep = stack.e[np] - RM;
                }

                /* Call ausgab and drop energy on spot */
                ausgab(edep);

                /* Positron annihilation section */
                if(iq > 0) {
                    /* The particle is a positron. Produce annihilation gammas
                    if edep < eie */
                    if(edep < eie) {
                        rannih();
                        
                        // Now discard the positron and take normal return to
                        // follow the annihilation gammas.
                        return;
                    }
                }
                
                stack.np -= 1;
                return;
			}
            
            if (ustep < 0) {
                /* Negative ustep */
                printf("Warning!, negative ustep = %f\n", ustep);
                ustep = 0.0;
            }
			double vstep;	// transport distance after truncation by howfar
			
			if (ustep == 0.0f || imed == -1) {
				/* Do fast step */
				if (ustep != 0.0f) {
					/* Step in vacuum */
					vstep = ustep;  // vstep is ustep truncated by howfar
					tvstep = vstep; // tvstep is the total curved path 
                                    // associated with vstep

					/* Transport the particle */
                    stack.x[np] += stack.u[np]*vstep;
                    stack.y[np] += stack.v[np]*vstep;
                    stack.z[np] += stack.w[np]*vstep;
                    stack.dnear[np] -= vstep;
				}   // end of vacuum step

				/* Electron region change */
                if(irnew != irold) {
                    stack.ir[np] = irnew;
                    irl = irnew;
				    imed = region.med[irl];
                }			

                /* First check of electron cut-off energy */
                if(eie <= region.ecut[irl]) {
                    
                    edep = stack.e[np] - RM;    // get energy deposition for user
                    
                    /* Call ausgab and drop energy on spot */
                    ausgab(edep);
                    
                    /* Positron annihilation section */
                    if(iq > 0) {
                        /* The particle is a positron. Produce annihilation 
                        gammas if edep < eie */
                        if(edep < eie) {
                            rannih();
                            
                            /* Now discard the positron and take normal return
                            to follow the annihilation gammas */
                            return;
                        }
                    }
                    
                    stack.np -= 1;
                    return;
                }	

				break;  // exit ustep loop. Go try another big step in 
                        // (possibly) new medium
			} 
            vstep = ustep;

			if (call_howfar) {
				/* We are in single scattering mode */
				tvstep = vstep;
				if (tvstep != tustep) {
					/* Boundary was crossed. Shut off single scattering */
					do_single = 0; // i.e. false
				}

				/* Fourth order technique for dedx. Must be done for a 
				single scattering step */
				de = computeEloss(imed, iq, irl, rhof, 
                    tvstep,	range, eke,	elke, lelke);
			}
			else {
				/* call_howfar = false. Step has not been reduced due to 
				boundaries */
				tvstep = tustep;
				if (called_msdist == 0) {
					/* Second order technique for dedx. Already done in a 
					normal CH step with call to msdist */
					de = computeEloss(imed,	iq, irl, rhof,
						tvstep,	range, eke, elke, lelke);
				}
			}   // end of call_howfar if-else sentence.
            
			edep = de;          // energy deposition variable for user
			ekef = eke - de;    // final kinetic energy
            
			/* Now do multiple scattering */
			double sinthe; double costhe; // deflection angle
			if (called_msdist == 0) { // everything done if called_msdist = true
				if (do_single) {    // Single scattering
                    /* kinetic energy used to sample MS angle 
                    (normally midpoint) */
					double ekems = fmax(ekef, region.ecut[irl] - RM);
					
                    p2 = ekems*(ekems + 2.0*RM);
					beta2 = p2/(p2 + pow(RM, 2.0));
					
                    /* Multiple scattering screening angle */
                    double chia2 = electron_data.xcc[imed]/
                        (4.0*electron_data.blcc[imed]*p2);

					/* We always consider spin effects */
					double elkems = log(ekems);
					int lelkems = pwlfInterval(imed, elkems, 
                        electron_data.eke1, electron_data.eke0) - 1;

					if (iq < 0) {
						etap = pwlfEval(MXEKE*imed+lelkems, elkems, 
                            electron_data.etae_ms1, electron_data.etae_ms0);
					}
					else {
						etap = pwlfEval(MXEKE*imed+lelkems, elkems, 
                            electron_data.etap_ms1, electron_data.etap_ms0);
					}
					chia2 *= etap;

					sscat(imed, qel, chia2,	elkems,	beta2, &costhe,	&sinthe);

				}
				else {
					// No deflection in single scattering mode.
					sinthe = 0.0f;
					costhe = 1.0f;
				}
			} // end of called_msdist if sentence

			/* We now know distance and amount of energy loss for this 
			step, and the scattering angle. Now is time to do the 
			transport */

			if (called_msdist == 0) {
				/* Calculate deflection and scattering. This has not been done 
                in msdist */
				x_final = stack.x[np] + stack.u[np]*vstep;
                y_final = stack.y[np] + stack.v[np]*vstep;
                z_final = stack.z[np] + stack.w[np]*vstep;

				if (do_single) {
					/* Apply the deflection, save call to uphi if no 
					deflection in a single scattering mode */
					uphi21(&uphi, costhe, sinthe);
					u_final = stack.u[np];
                    v_final = stack.v[np];
                    w_final = stack.w[np];
				}
				else {
					u_final = stack.u[np];
                    v_final = stack.v[np];
                    w_final = stack.w[np];
				}
			}

            /* The electron step is about to occur, score the energy 
            deposited */
            ausgab(edep);

			/* Transport the particle */
			stack.x[np] = x_final;
            stack.y[np] = y_final;
            stack.z[np] = z_final;
			stack.u[np] = u_final;
            stack.v[np] = v_final;
            stack.w[np] = w_final;
			stack.dnear[np] -= vstep;
			irold = stack.ir[np];		// save the previous region

			/* Now done with multiple scattering, update energy and see if 
			below cut below substracts only energy deposited */
			eie -= edep;
			stack.e[np] = eie;

            if(irnew == irl && eie <= region.ecut[irl]) {
                    
                    edep = stack.e[np] - RM;    // get energy deposition for user
                    
                    /* Call ausgab and drop energy on spot */
                    ausgab(edep);
                    
                    /* Positron annihilation section */
                    if(iq > 0) {
                        /* The particle is a positron. Produce annihilation gammas
                        if edep < eie */
                        if(edep < eie) {
                            rannih();
                            
                            // Now discard the positron and take normal return to
                            // follow the annihilation gammas.
                            return;
                        }
                    }
                    
                    stack.np -= 1;
                    return;
                }

			medold = imed;  // save previous region
			if (imed != -1) {
				eke = eie - RM; // update kinetic energy
				elke = log(eke);

				/* Get updated interval */
				lelke = pwlfInterval(imed, elke, 
                        electron_data.eke1, electron_data.eke0) - 1;
			}

			/* Electron region change */
            if(irnew != irold) {
                stack.ir[np] = irnew;
                irl = irnew;
                imed = region.med[irl];
            }
			
            /* Check electron cut-off energy */
			if(eie <= region.ecut[irl]) {
        
                edep = stack.e[np] - RM;    // get energy deposition for user
                
                /* Call ausgab and drop energy on spot */
                ausgab(edep);
                
                /* Positron annihilation section */
                if(iq > 0) {
                    /* The particle is a positron. Produce annihilation gammas
                    if edep < eie */
                    if(edep < eie) {
                        rannih();
                        
                        // Now discard the positron and take normal return to
                        // follow the annihilation gammas.
                        return;
                    }
                }
                
                stack.np -= 1;
                return;
            }

			if (imed != medold) {
				break;  // exit to tstep loop
			}

			/* Update demfp. As energy loss is used as the 'path-length' 
			variable, it just substracts the energy loss for the step */
			demfp -= de*sig0; 
			total_de -= de;
			total_tstep -= tvstep*rhof;
			if (total_tstep < 1.0E-9) {
				demfp = 0.0;
			}

        } while (demfp >= EPSEMFP); // end of ustep loop

        /* If following is true, it means that particle has carried out a fast 
    	step in vacuum or has changed medium. In that case, go to beginning of 
		tstep loop */
		
		if ((imed != medold) || (ustep == 0.0) || (imed == -1)) {
			continue; // start at beginning of tstep loop
		}

		/* Compute final sigma to see if resample is needed. This will take 
		the energy variation of the sigma into account using the 
		fictitious sigma method */
		if (iq < 0) {
			sigf = pwlfEval(imed*MXEKE+lelke, elke, 
                electron_data.esig1, electron_data.esig0);
			dedx0 = pwlfEval(imed*MXEKE+lelke, elke, 
                electron_data.ededx1, electron_data.ededx0);
			sigf /= dedx0;
		}
		else {
			sigf = pwlfEval(imed*MXEKE+lelke, elke, 
                electron_data.psig1, electron_data.psig0);
			dedx0 = pwlfEval(imed*MXEKE+lelke, elke, 
                electron_data.pdedx1, electron_data.pdedx0);
			sigf /= dedx0;
		}
		
		sigratio = sigf/sig0;
		rfict = setRandom();

    } while (rfict >= sigratio);    // end of tstep loop
        
    /* Now sample electron interaction */
    if (iq < 0) {
		/* electron. Check branching ratio */
		double ebr1 = pwlfEval(imed*MXEKE+lelke, elke,  // e- branching ratio 
            electron_data.ebr11, electron_data.ebr10);  // into brem
		rnno = setRandom();
		if (rnno <= ebr1) {
			/* It was Bremsstrahlung */
			brems();            
		}
		else {
			/* It was Moller, but first check the kinematics. However, if 
			EII is on we should still permit an interaction, even if 
			E < Moller threashold as EII interactions go down to the 
			ionization threshold which may be less than thmoll */
			if (stack.e[np] <= pegs_data.thmoll[imed]) {
				/* Not enough energy for Moller, so force it to be a 
				Bremsstrahlung, provided ok kinematically */

				if (ebr1 <= 0) {    // Brems not allowed either.
                    /* Return to shower to re-enter electron() */
					return;
				}
				else {
					brems();
				}
			}
			else {
				moller();
			}
		}
	}
	else {
		/* Positron interaction. pbr1 = brems/(brems + bhabha + annih) */
		double pbr1 = pwlfEval(imed*MXEKE+lelke, elke,  // e+ branching ratio
            electron_data.pbr11, electron_data.pbr10);	// into brem.
		rnno = setRandom();
		if (rnno < pbr1) {
			/* It was bremsstrahlung */
			brems();
		}
		else {
			/* Decide between bhabha and annihilation. 
			pbr2 = (brems + bhabha)/(brems + bhabha + annih) */
			double pbr2 = pwlfEval(imed*MXEKE+lelke, elke,  //e+ branching ratio
                electron_data.pbr21, electron_data.pbr20);  // into brem or Bha.	
			if (rnno < pbr2) {
				/* It is bhabha */
				bhabha();
			}
			else {
				/* It is in-flight annihilation */
                annih();
			}   // end pbr2 if-else sentence		
		}
	}
    
    /* Return to shower */
    return;
}

double computeDrange(int imed, int iq, int lelke, double ekei, double ekef,
                     double elkei, double elkef) {
    /* The following function computes the path-length traveled while going from
	energy ekei to energy ekef, both energies being in the same
	interval of the PWL function, whose index is given by lekle. 
	elkei and elkef are the logarithms of ekei and ekef */

	/* This function is based on logarithmic interpolation as
	used in EGSnrc (i.e. dedx = a + b*Log(E) ) and a power series expansion
	of the ExpIntegralEi function that is the result of the integration */

	double dedxmid; 
    double aux;
	double fedep = 1.0 - ekef/ekei;

	/* First evaluate the logarithm of the energy midpoint */
	double elktmp = 0.5*(elkei + elkef +
		0.25*pow(fedep, 2.0)*(1.0 + fedep*(1.0 + 0.875*fedep)));

	if (iq < 0) {
		dedxmid = pwlfEval(MXEKE*imed+lelke, elktmp, 
            electron_data.ededx1, electron_data.ededx0);
		dedxmid = 1.0/dedxmid;
		aux = electron_data.ededx1[MXEKE*imed+lelke]*dedxmid;
	}
	else {
		dedxmid = pwlfEval(MXEKE*imed+lelke, elktmp, 
            electron_data.pdedx1, electron_data.pdedx0);
		dedxmid = 1.0/dedxmid;
		aux = electron_data.pdedx1[MXEKE*imed+lelke]*dedxmid;
	}

	aux = aux*(1.0 + 2.0*aux)*pow(fedep/(2.0 - fedep),2.0)/6.0;

	return fedep*ekei*dedxmid*(1.0 + aux);
}

double computeEloss(int imed, int iq, int irl, double rhof, 
    double tustep, double range, double eke, double elke, int lelke) {

    /* This function computes the energy loss due to sub-threshold
	processes for a path-length tustep. The energy at the beginning of
	the step is eke, elke = Log(eke), lelke is the interval index for 
	the PWLF interpolation */
	double aux = 0.0;
	double dedxmid;
	double de;
	double fedep;
    double elktmp;
    double eketmp;
    int lelktmp;
	int qel = (1+iq)/2;

	/* Calculate the range between the initial energy and the next lower energy 
	on the interpolation grid */
	double tuss = range - 
        electron_data.range_ep[qel*geometry.nmed*MXEKE+imed*MXEKE+lelke]/rhof;

	if (tuss >= tustep) {
		/* Final energy is in the same interpolation bin. Use the EGSnrc 
		logarithmic interpolation method */

		if (iq < 0) {
			dedxmid = pwlfEval(imed*MXEKE+lelke, elke, 
                electron_data.ededx1, electron_data.ededx0);
			aux = electron_data.ededx1[imed*MXEKE+lelke]/dedxmid;
		}
		else {
			dedxmid = pwlfEval(imed*MXEKE+lelke, elke, 
                electron_data.pdedx1, electron_data.pdedx0);
			aux = electron_data.pdedx1[imed*MXEKE+lelke]/dedxmid;
		}

		de = dedxmid*tustep*rhof;
		fedep = de/eke;
		de *= (1.0 - 0.5*fedep*aux*(1.0 - 0.333333*fedep*(aux - 1.0 - 
            0.25*fedep*(2.0 - aux*(4.0 - aux)))));
	}
	else {
		/* Must find first the table index where the step ends using 
		pre-calculated ranges */
		lelktmp = lelke;

		/* now tuss is the range of the final energy 
		electron scaled to the default mass density 
		from PEGS4 */
		tuss = (range - tustep)*rhof;

		if (tuss <= 0) {
			de = eke - pegs_data.te[imed]*0.99;
		}
		/* i.e., if the step we intend to take is longer than the particle 
        range, the particle energy goes down to the threshold (eke is the 
        initial particle energy) */

		/* Originally the entire energy was lost, but msdist is not prepared to 
        deal with such large eloss fractions */
		else {
			while (tuss < electron_data.range_ep[qel*geometry.nmed*MXEKE+
                    imed*MXEKE+lelktmp]) {
				lelktmp -= 1;
			}
			elktmp = (lelktmp+2 - electron_data.eke0[imed])/
                electron_data.eke1[imed];
			eketmp = electron_data.e_array[imed*MXEKE+lelktmp+1];

			tuss = (electron_data.range_ep[qel*geometry.nmed*MXEKE+imed*MXEKE
                +lelktmp+1]-tuss)/rhof;

			if (iq < 0) {
				dedxmid = pwlfEval(MXEKE*imed+lelktmp, elktmp, 
                    electron_data.ededx1, electron_data.ededx0);
				aux = electron_data.ededx1[MXEKE*imed+lelktmp]/dedxmid;
			}
			else {
				dedxmid = pwlfEval(MXEKE*imed+lelktmp, elktmp, 
                    electron_data.pdedx1, electron_data.pdedx0);
				aux = electron_data.pdedx1[MXEKE*imed+lelktmp]/dedxmid;
			}
			de = dedxmid*tuss*rhof;
			fedep = de / eketmp;
			de *= (1.0 - 0.5*fedep*aux*(1.0 - 0.333333*fedep*(aux - 1.0 - 
                0.25*fedep*(2.0 - aux*(4.0 - aux)))));

			de += eke - eketmp;
		}
	}

	return de;
}

double msdist(int imed, int iq, double rhof, double de, double tustep, 
    double eke, double *x_final, double *y_final, double *z_final, 
    double *u_final, double *v_final, double *w_final) {
    
    int qel = (1 + iq)/2;   // = 0 for electrons, = 1 for positrons
    
    /* The following auxiliary variables must persist among calls to mscat and 
    related functions */
    struct Mscats m_scat;
    struct Spinr spin_r;

    double blccc = electron_data.blcc[imed];
    double xcccc = electron_data.xcc[imed];

    /* Commonly used factors */
    double e = eke - 0.5*de;
	double tau = e/RM;  // average kinetic energy over the step divided by 
                        // electron mass
	double tau2 = pow(tau, 2.0);
	double epsilon = de/eke;    // fractional energy loss
	double epsilonp = de/e;

	e *= (1.0 - pow(epsilonp, 2.0)*(6.0 + 10.0*tau + 5.0*tau2)/(24.0*tau2 + 
        72.0*tau + 48.0));

	double p2 = e*(e + 2.0*RM); // average momentum over the step
	double beta2 = p2/(p2 + pow(RM, 2.0));  // speed at e in units of c, squared
	double chia2 = xcccc/(4.0*p2*blccc);    // screening angle, note: our chia2 
										    // is Moliere's chia2/4
    
	double lambda = 0.5*tustep*rhof*blccc/beta2;    // distance in number of 
                                                    // elastic scattering mean 
                                                    // free paths for each 
                                                    // sample of the multiple 
                                                    // scattering angle

	double temp2 = 0.166666*(4.0 + tau*(6.0 + tau*(7.0 + tau*(4.0 + tau))))*
        (epsilonp/((tau + 1.0)*(tau + 2.0)))*
        (epsilonp/((tau + 1.0)*(tau + 2.0)));   // auxiliary variable for energy
                                                // loss corrections
	lambda *= (1.0 - temp2);

	double elke = log(e);
	int lelke = pwlfInterval(imed, elke,    // adjusted to C index standard
        electron_data.eke1, electron_data.eke0) - 1;

    if (lelke < 0) {    // This should normally not happen
		lelke = 0;
		elke = (1.0 - electron_data.eke0[imed])/electron_data.eke1[imed];
	}

    double etap;    // correction to the screening parameter derived from PWA
	double xi_corr; // correction to the first MS moments due to spin
	double gamma;   // q2/q1

	if (qel == 0) {
		etap = pwlfEval(MXEKE*imed+lelke, elke, 
            electron_data.etae_ms1, electron_data.etae_ms0);
		xi_corr = pwlfEval(MXEKE*imed+lelke, elke, 
            electron_data.q1ce_ms1, electron_data.q1ce_ms0);
		gamma = pwlfEval(MXEKE*imed+lelke, elke, electron_data.q2ce_ms1, 
            electron_data.q2ce_ms0);
	}
	else {
		etap = pwlfEval(MXEKE*imed+lelke, elke, 
            electron_data.etap_ms1, electron_data.etap_ms0);
		xi_corr = pwlfEval(MXEKE*imed+lelke, elke, 
            electron_data.q1cp_ms1, electron_data.q1cp_ms0);
		gamma = pwlfEval(MXEKE*imed+lelke, elke, 
            electron_data.q2cp_ms1, electron_data.q2cp_ms0);
	}

    double ms_corr = pwlfEval(MXEKE*imed+lelke, elke, 
        electron_data.blcce1, electron_data.blcce0);    // correction to the 
                                                        // first MS moments due 
                                                        // to spin
	chia2 *= etap;

    lambda /= (etap*(1.0 + chia2));
	lambda *= ms_corr;
	double chilog = log(1.0 + 1.0/chia2);
	double q1 = 2.0*chia2*(chilog*(1.0 + chia2) - 1.0);	// first moment of the 
                                                        // single scattering 
                                                        // cross section

	gamma = 6.0*chia2*(1.0 + chia2)*(chilog*(1.0 + 2.0*chia2) - 2.0)/q1*gamma;
	double xi = q1*lambda; // first GS - moment

    /* Sample first substep scattering angle */
	int find_index = 1; // i.e. true
	int spin_index = 1;
	double w1;      // cosine of the first substep polar scattering angle
	double sint1;   // sine of the first substep polar scattering angle
	mscat(imed, qel, &spin_index, &find_index, elke, beta2, xi, 
		lambda,	chia2,	&w1, &sint1, &m_scat, &spin_r);

	double cphi1;   // sine and cosine of the first azimuthal angle 
    double sphi1;
	double phi = 2.0*M_PI*setRandom();
    cphi1 = cos(phi);
    sphi1 = sin(phi);
    
    /* Sample second substep scattering angle */
	double w2;      // cosine of the second substep polar scattering angle
	double sint2;   // sine of the second substep polar scattering angle
	mscat(imed, qel, &spin_index, &find_index, elke, beta2, xi, 
		lambda,	chia2,	&w2, &sint2, &m_scat, &spin_r);

	double cphi2;   // sine and cosine of the second azimuthal angle 
    double sphi2;
	phi = 2.0*M_PI*setRandom();
    cphi2 = cos(phi);
	sphi2 = sin(phi);

    /* Final direction of motion, relative to z-axis */
	double u2 = sint2*cphi2;
	double v2 = sint2*sphi2;
	double u2p = w1*u2 + sint1*w2;

    /* Direction cosines after scattering */
    double us = u2p*cphi1 - v2*sphi1;
    double vs = u2p*sphi1 + v2*cphi1;
    double ws = w1*w2 - sint1*u2;
	
    /* Calculate delta, b, c */
    xi *= 2*xi_corr;

    double eta = setRandom();   // randomization of substep transport distances
    double eta1 = 0.5*(1.0 - eta);
    double delta = 0.9082483 - (0.1020621 - 0.0263747*gamma)*xi;

    double temp1 = 2.0 + tau;   // auxilarity variables for energy 
							    // loss corrections
	double temp = (2.0 + tau*temp1)/((tau + 1.0)*temp1);

	/* Take logarithmic dependence into account as well */
	temp -= (tau + 1.0)/((tau + 2.0)*(chilog*(1.0 + chia2) - 1.0));
	temp *= epsilonp;
	temp1 = 1.0 - temp;
	delta += 0.40824829*(epsilon*(tau + 1.0)/((tau + 2.0)*(chilog*(1.0 + 
        chia2) - 1.0)*(chilog*(1.0 + 2.0*chia2) - 2.0))	- 0.25*pow(temp, 2.0));
	double b = eta*delta;           // substep transport distance
	double c = eta*(1.0 - delta);   // substep transport distance	

	/* Calculate transport direction cosines */
	double w1v2 = w1 * v2;
    double ut = b*sint1*cphi1 + c*(cphi1*u2 - sphi1*w1v2) + eta1*us*temp1;
    double vt = b*sint1*sphi1 + c*(sphi1*u2 + cphi1*w1v2) + eta1*vs*temp1;
    double wt = eta1*(1.0 + temp) + b*w1 + c*w2 + eta1*ws*temp1;

	/* Calculate transport distance */
	double ustep = tustep*sqrt(pow(ut, 2.0) + pow(vt, 2.0) + pow(wt, 2.0));

	/* Rotate into the final direction of motion and transport relative to 
    original direction of motion */
    int np = stack.np;
    double x0 = stack.x[np];
    double y0 = stack.y[np];
    double z0 = stack.z[np];
    double u0 = stack.u[np];
    double v0 = stack.v[np];
    double w0 = stack.w[np];    
	double sint02 = pow(u0, 2.0) + pow(v0, 2.0);

    if (sint02 > 1.0E-20) {
		double sint0  = sqrt(sint02);
        double sint0i = 1.0/sint0;
        double cphi0  = sint0i*u0;
        double sphi0  = sint0i*v0;

        /* Scattering angles */
        u2p = w0*us + sint0*ws;
        ws = w0*ws - sint0*us;
        us = u2p*cphi0 - vs*sphi0;
        vs = u2p*sphi0 + vs*cphi0;

        /* Transport angles */
        u2p = w0*ut + sint0*wt;
        wt = w0*wt - sint0*ut;
        ut = u2p*cphi0 - vt*sphi0;
        vt = u2p*sphi0 + vt*cphi0;
	}
	else {
        wt = w0*wt; ws = w0*ws;
	}

    /* Transport the particle. Transfer new position and direction */
    *x_final = x0 + tustep*ut;
    *y_final = y0 + tustep*vt;
    *z_final = z0 + tustep*wt;
    *u_final = us;
    *v_final = vs;
    *w_final = ws;
    
    return ustep;
}

void mscat(int imed, int qel, int *spin_index, int *find_index, 
    double elke, double beta2, double q1,  double lambda, double chia2, 
    double *cost, double *sint, struct Mscats *m_scat, struct Spinr *spin_r) {
    /* Function to sample multiple electron scattering angles from the exact 
	distribution resulting from elastic scattering, described by the 
	screened Rutherford cross times Mott correction (i.e. spin_effects=true) */
	double xi, rejf, rnno;
	double explambda = exp(-lambda);

    if (lambda <= 13.8) {
		/* Test only for lambda = 13.8 implies a 1E-6 error, i.e. large 
		lambda cases that contribute to the forward no-scattering amplitude */
		double sprob = setRandom();
		if (sprob < explambda) {
			/* It was a no scattering event */
			*cost = 1.0;
			*sint = 0.0;
			return;
		}

		double wsum = (1.0 + lambda)*explambda;
		if (sprob <  wsum) {
			do {
				xi = setRandom();
				xi = 2.0*chia2*xi/(1.0 - xi + chia2);
				*cost = 1.0 - xi;

				rejf = spinRejection(imed, qel,	elke, beta2, q1,
					*cost, spin_index, 0, spin_r);
				rnno = setRandom();
			} while (rnno > rejf);

			*sint = sqrt(xi*(2.0 - xi));
			return;
		}

		if (lambda <= 1) {
			int icount = 0;
			double wprob = explambda;
			double sinz, cosz;
			double phi;
			wsum = explambda;
			*cost = 1.0;
			*sint = 0.0;

			do {
				icount += 1;
				if (icount > 20) {
					break;
				}   // to avoid underflow if sprob very close to 1

				wprob = wprob * lambda / icount;
				wsum = wsum + wprob;
				do {
					/* the following applies to the case where spin effects are 
					enabled */
					xi = setRandom();
					xi = 2.0*chia2*xi/(1.0 - xi + chia2);
					cosz = 1.0 - xi;

					rejf = spinRejection(imed,	qel, elke, beta2, q1,
						cosz, spin_index, 0, spin_r);
					rnno = setRandom();
				} while (rnno > rejf);

				sinz = xi * (2.0 - xi);
				if (sinz > 1.0E-20) {
					sinz = sqrt(sinz);
					xi = setRandom();
					phi = xi*6.2831853;
					*cost = (*cost)*cosz - *sint*sinz*cos(phi);
					*sint = sqrt(fmax(0.0, (double)((1.0  - 
                        (*cost))*(1.0 + (*cost)))));
				}
			} while (wsum <= sprob);
			return;
		}
	}

	/* It was a multiple scattering event. Sample the angle from the q^(2+) 
	surface */
	if (lambda <= LAMBMAX_MS) {
		double ai, aj;
		double llmbda = log(lambda);

		if (*find_index) {
			/* First find lambda bin */
			ai = llmbda*mscat_data.dllambi;
			m_scat->i = (int)ai;
			ai -= (double)m_scat->i;
			xi = setRandom();

			if (xi < ai) {
				m_scat->i += 1;
			}

			if (q1 < QMIN_MS) {
				m_scat->j = 0;
			}
			else if (q1 < QMAX_MS) {
				aj = q1*mscat_data.dqmsi;
				m_scat->j = (int)aj;
				aj -= (double)m_scat->j;
				xi = setRandom();
				if (xi < aj) {
					m_scat->j += 1;
				}
			}
			else {
				m_scat->j = MXQ_MS;
			}

			/* Calculate omega2 */
			if (llmbda < 2.2299) {
				m_scat->omega2 = chia2*(lambda + 4.0)*(1.347006	+ 
                    llmbda*(0.209364 - llmbda*(0.45525 - llmbda*(0.50142 - 
                    0.081234*llmbda))));
			}
			else {
				m_scat->omega2 = chia2*(lambda + 4.0)*(-2.77164	+ 
                    llmbda*(2.94874 - llmbda*(0.1535754	- llmbda*0.00552888)));

			}
			*find_index = 0;    // i.e. false
		}

		/* If this is a re-iteration with the same lambda, then omega2, i, k 
		should be defined in the previous section */
		int counter2 = 0;
		int k;
		double a; double ak; double u; double du; double x1;
		do {
			counter2++;
			xi = setRandom();
			ak = xi*MXU_MS;
			k = ak;
			ak -= k;

			if (ak > mscat_data.wms_array[m_scat->i + (MXL_MS+1)*(m_scat->j) + 
                (MXL_MS+1)*(MXQ_MS+1)*k]) {
				k = mscat_data.ims_array[m_scat->i + (MXL_MS+1)*(m_scat->j)	+ 
                    (MXL_MS+1)*(MXQ_MS+1)*k];
			}

			a = mscat_data.fms_array[m_scat->i + (MXL_MS+1)*(m_scat->j)	+ 
                (MXL_MS+1)*(MXQ_MS+1)*k];
			u = mscat_data.ums_array[m_scat->i + (MXL_MS+1)*(m_scat->j) + 
                (MXL_MS+1)*(MXQ_MS+1)*k];
			du = mscat_data.ums_array[m_scat->i + (MXL_MS+1)*(m_scat->j) + 
                (MXL_MS+1)*(MXQ_MS+1)*(k+1)] - u;
			xi = setRandom();

			if (fabs(a) < 0.2) {
				x1 = 0.5*(1.0 - xi)*a;
				u += xi*du*(1.0 + x1*(1.0 - xi*a));
			}
			else {
				u -= du/a*(1.0 - sqrt(1.0 + xi*a*(2.0 + a)));
			}

			xi = m_scat->omega2*u/(1.0 + 0.5*m_scat->omega2 - u);
			if (xi > 1.99999) {
				xi = 1.99999;
			}

			/* Some machines have trouble when xi is very close to 2 
			in subsequent calculations */
			*cost = 1.0 - xi;

			rejf = spinRejection(imed, qel,	elke, beta2, q1,
				*cost, spin_index, 0, spin_r);

			rnno = setRandom();
		} while (rnno > rejf);

		*sint = sqrt(xi*(2.0 - xi));
	}

    return;    
}

double spinRejection(int imed, int qel,	double elke, double beta2, double q1,
	double cost, int *spin_index, int is_single, struct Spinr *spin_r) {
	
    /* This function determines the rejection function due to spin effects */
	int k;
	double ai; double aj; double ak; double qq1; double xi;
	double rnno;

	if (*spin_index) {
		/* Determine the energy and q1 index */
		*spin_index = 0;    // i.e. false

		if (beta2 >= spin_data.b2spin_min) {
			ai = (beta2 - spin_data.b2spin_min)*spin_data.dbeta2i;
			spin_r->i = (int)ai;
			ai -= (double)spin_r->i;
			spin_r->i += MXE_SPIN + 1;
		}
		else if (elke > spin_data.espml) {
			ai = (elke - spin_data.espml)*spin_data.dleneri;
			spin_r->i = (int)ai;
			ai -= spin_r->i;
		}
		else {
			spin_r->i = 0;
			ai = -1.0f;
		}

		rnno = setRandom();
		if (rnno < ai) {
			spin_r->i += 1;
		}

		if (is_single) {
			spin_r->j = 0;
		}
		else {
			qq1 = 2.0*q1;
			qq1 = qq1/(1.0 + qq1);
			aj = qq1*spin_data.dqq1i;

			spin_r->j = (int)aj;
			if (spin_r->j >= MXQ_SPIN) {
				spin_r->j = MXQ_SPIN;
			}
			else {
				aj -= (double)spin_r->j;
				rnno = setRandom();
				if (rnno < aj) {
					spin_r->j += 1;
				}
			}
		}
	}

	xi = sqrt(0.5*(1.0 - cost));
	ak = xi*MXU_SPIN;
	k = (int)ak;
	ak -= (double)k;

	double spin_rej = spin_data.spin_rej[
        imed*2*(MXE_SPIN1+1)*(MXQ_SPIN+1)*(MXU_SPIN+1) + 
        qel*(MXE_SPIN1+1)*(MXQ_SPIN+1)*(MXU_SPIN+1) + 
        spin_r->i*(MXQ_SPIN+1)*(MXU_SPIN+1) + spin_r->j*(MXU_SPIN+1) + k];
	double spin_rej2 =	spin_data.spin_rej[
        imed*2*(MXE_SPIN1+1)*(MXQ_SPIN+1)*(MXU_SPIN+1) + 
        qel*(MXE_SPIN1+1)*(MXQ_SPIN+1)*(MXU_SPIN+1) + 
        spin_r->i*(MXQ_SPIN+1)*(MXU_SPIN+1) + spin_r->j*(MXU_SPIN+1) + (k+1)];
	double spin_reject = (1.0 - ak)*(spin_rej) + ak*spin_rej2;

	return spin_reject;
}

void sscat(int imed, int qel, double chia2, double elke, double beta2,
	double *cost, double *sint) {
	/* single elastic scattering */

	/* The following auxiliary variables must persist among calls to 
	spin_rejection */
	int spin_index = 1; // i.e. true
	struct Spinr spin_r;

	int qzero;
	double xi; double rejf; double rnno;

	do {
		xi = setRandom();
		xi = 2.0*chia2*xi/(1.0 - xi + chia2);
		*cost = 1.0 - xi;

		/* We always consider spin effects turned on */
		qzero = 0;

		rejf = spinRejection(imed, qel, elke, beta2, qzero,	
            *cost,	&spin_index, 1,	&spin_r);

		rnno = setRandom();
	} while (rnno > rejf);

	*sint = sqrt(xi*(2.0 - xi));

	return;
}

void rannih() {
    
    /* Pick random direction for first gamma */
    double rnno;
    int np = stack.np;
    
    /* Polar angle selection */
    rnno = setRandom();
    double costhe = 2.0*rnno - 1;
    double sinthe = sqrt(fmax(0.0, (1.0 - costhe)*(1.0 + costhe)));
    
    
    /* Azimuthal angle selection */
    rnno = setRandom();
    double phi = 2.0*M_PI*rnno;
    double sinphi = sin(phi);
    double cosphi = cos(phi);
    
    /* First photon */
    stack.e[np] = RM;
    stack.iq[np] = 0;
    stack.u[np] = sinthe*cosphi;
    stack.v[np] = sinthe*sinphi;
    stack.w[np] = costhe;
    
    /* Second photon */
    np +=1;
    stack.e[np] = RM;
    stack.iq[np] = 0;
    transferProperties(np, np-1);
    stack.u[np] = -1.0*stack.u[np-1];
    stack.v[np] = -1.0*stack.v[np-1];
    stack.w[np] = -1.0*stack.w[np-1];
    
    /* Update stack */
    stack.np = np;
    
    return;
}

void brems() {
    /* This function samples Bremmsstrahlung energy using Coulomb corrected 
	Bethe-Heitler above 50 MeV and Bethe-Heitler below 50 MeV. This option 
	corresponds to ibr_nist = 0 in the EGSnrc platform */

	int np = stack.np;	
	int irl = stack.ir[np];
	int imed = region.med[irl];
	
    double eie = stack.e[np];   // energy of incident electron
	double phi1; double phi2;   // screening function
	
	/* Decide which distribution to use:
	Coulomb corrected Bethe-Heitler above 50 MeV
	Bethe-Heitler elsewhere */
	int l;
	if(eie < 50.0) { 
		l = 1; // BH
	}
	else { 
		l = 3; // BH Coulomb corrected
	}
	int l1 = l + 1;

	double ekin = eie - RM; // kinetic incident energy
	double brmin = pegs_data.ap[imed]/ekin;
	double waux = -log(brmin);

	double a; double b; double c;   // direction cosines of incident electron.
	double sinpsi; double sindel; double cosdel;    // all used for rotations.
	double ztarg;   // (Zeff^1/3/111)^2, used for 2BS angle sampling
	double tteie;   // total energy in units of rest energy
	double y2maxi;
	double z2maxi;

	// We will sample the photon emmision angle from KM-2BS (ibrdst=1) or 
    // from the leading term (ibrdst=0).
    a = stack.u[np];
    b = stack.v[np];
    c = stack.w[np];

    sinpsi = pow(a, 2.0) + pow(b, 2.0);
    if(sinpsi > 1.0E-20) {
        sinpsi = sqrt(sinpsi);
        sindel = b/sinpsi;
        cosdel = a/sinpsi;
    }

    ztarg = pair_data.zbrang[imed];
    tteie = eie/RM;
    /* Electron velocity in speed of light units */
    double beta = sqrt((tteie - 1.0)*(tteie + 1.0))/tteie;

    double y2max = 2.0*beta*(1.0 + beta)*tteie*tteie;   // maximum possible 
                                                        // scaled angle
    y2maxi = 1.0/y2max;
    double z2max = y2max + 1.0;
    z2maxi = sqrt(z2max);

	/* We do not implement Bremsstrahlung splitting */

	double aux;
	double br;                      // energy fraction of secondary photon
	double delta;                   // scaled momentum transfer
	double rnno06; double rnno07;   // random numbers
	double rejf;                    // screening rejection function
	double ese;                     // total energy of scattered electron
	double esg;                     // energy of emitted photon

	do { 
		rnno06 = setRandom();
		rnno07 = setRandom();
		br = brmin*exp(rnno06*waux);
		esg = ekin*br; 
		ese = eie - esg;
		delta = esg/eie/ese*pair_data.delcm[imed]; 
		aux = ese/eie;

		if( delta < 1.0 ) {
			phi1 = pair_data.dl1[imed*8+l-1] + delta*(pair_data.dl2[imed*8+l-1]+ 
                delta*pair_data.dl3[imed*8+l-1]);
			phi2 = pair_data.dl1[imed*8+l1-1]+delta*(pair_data.dl2[imed*8+l1-1]+
					delta*pair_data.dl3[imed*8+l1-1]);
		}
		else {
			phi1 = pair_data.dl4[imed*8+l-1] + pair_data.dl5[imed*8+l-1]*
                log(delta + pair_data.dl6[imed*8+l-1]);
			phi2 = phi1;
		}
		rejf = (1.0 + pow(aux, 2.0))*phi1 - 2.0*aux*phi2/3.0;		
	} while(rnno07 >= rejf);

	/* Setup the new photon */
	np += 1;
	stack.e[np] = esg;
	stack.iq[np] = 0;
	transferProperties(np, np-1);

	/* Now we need to decide the direction of the photon */
    double y2tst;                   // scaled angle, costhe = 1 - 2*y2tst/y2max
    double ttese = ese/RM;          // new electron energy in units of RM
    double esedei = ttese/tteie;    // new total energy over old total energy
                                    // used for angle rejection function calls
    double rejmax;
    double rjarg1 = 1.0 + esedei*esedei; 
    double rjarg2 = rjarg1 + 2.0*esedei;
    double rjarg3; 

    double rtest = 1.0; double rejtst= 0.0;
    aux = 2.0*ese*tteie/esg;
    aux = pow(aux, 2.0);
    double aux1 = aux*ztarg;

    if(aux1 > 10.0) {
        rjarg3 = -log(pair_data.zbrang[imed]) + (1.0 - aux1)/pow(aux1, 2.0);
    }
    else {
        rjarg3 = log(aux/(1.0 + aux1));
    }
    rejmax = rjarg1*rjarg3 - rjarg2;

    while(rtest >= rejtst) {
        y2tst = setRandom();
        rtest = setRandom();
        double aux3 = z2maxi/(y2tst + (1.0 - y2tst)*z2maxi);
        rtest = rtest*aux3*rejmax;
        y2tst = pow(aux3, 2.0) - 1.0;
        double y2tst1 = esedei*y2tst/pow(aux3, 4.0);
        double aux4 = 16.0*y2tst1 - rjarg2;
        double aux5 = rjarg1 - 4.0*y2tst1;

        if (rtest < aux4 + aux5*rjarg3){
            break;
        }

        double aux2 = log(aux/(1.0 + aux1/pow(aux3, 4.0)));
        rejtst = aux4 + aux5*aux2;
    }

	double costhe = 1.0 - 2.0*y2tst*y2maxi;
    double sinthe = sqrt(fmax(0.0 ,(1.0 - pow(costhe, 2.0))));

    /* Azimuthal angle sampling */
    double phi = 2.0*setRandom()*M_PI;
    double cphi = cos(phi);
    double sphi = sin(phi);

    if( sinpsi >= 1.0E-10 ) {
        double us = sinthe*cphi;
        double vs = sinthe*sphi;

        stack.u[np] = c*cosdel*us - sindel*vs + a*costhe;
        stack.v[np] = c*sindel*us + cosdel*vs + b*costhe;
        stack.w[np] = c*costhe - sinpsi*us;
    }
    else {
        stack.u[np] = sinthe*cphi;
        stack.v[np] = sinthe*sphi;
        stack.w[np] = c*costhe;
    }

	/* Set energy of the electron */
	stack.e[np-1] = ese;
	
	/* Update stack index */
	stack.np = np;

    return;
}

void moller() {

    /* In this function we sample moller scattering, for this process to happen 
	we need a minimum energy, if we don't have that energy the electron
	is in a continual energy loss, the theory applied uses conservation of 
	4-momentum and the moller formula for the cross section (James Bjorken, 
	Sidney Drell: Relativistische Quantenmechanik (Relativistic quantum 
	mechanics E. Akademischer Verlag Spektrum, Heidelberg 1998) */

    int np = stack.np;
    int irl = stack.ir[np];
    int imed = region.med[irl];
    double eie = stack.e[np];   // total energy of incident electron
	double ekin = eie - RM;	    // kinetic energy of incident electron

    if(ekin <= 2.0*pegs_data.te[imed]) { 
		/* Kinetic energy threshold not reached, therefore a Moller scattering
		cannot happen */
		return;
	}

	double t0 = ekin/RM;    // kinetic energy of incident electron in RM units
	double e0 = t0 + 1.0;   // total energy of incident electron in RM units
	double extrae = eie - pegs_data.thmoll[imed]; // energy above Moller thresh

	double g2; double g3;   // used for rejection function calculation
	g2 = pow(t0, 2.0)/pow(e0, 2.0); 
	g3 = (2.0*t0 + 1.0)/pow(e0, 2.0);
	
	double br;  // kinetic energy fraction to lowew energy electron
	double gmax = (1.0 + 1.25*g2);  // maximum value of the rejection function
	double rejf4;   // rejection function
	double r;
	double rnno27; double rnno28;   // random numbers

	do {
		/* To retry if rejected */
		rnno27 = setRandom();

        /* set epsilon, which is the ratio between the scattered kinetic energy 
        and the kinetic incident energy */
		br = pegs_data.te[imed]/(ekin - extrae*rnno27); 
		r = br/(1.0 - br);
		rnno28 = setRandom();
		rejf4 = (1.0 + g2*pow(br, 2.0) + r*(r - g3)); // rejection function 
													  // multiplied by gmax
		rnno28 *= gmax;
	} while(rnno28 > rejf4);

	double ekse2 = br*ekin;     // kinetic energy of secondary electron #2
	double ese1 = eie - ekse2;  // energy of secondary electron #1
	double ese2 = ekse2 + RM;   // energy of secondary electron #2

	stack.e[np] = ese1;
	stack.e[np+1] = ese2;

	double h1 = (eie + RM)/ekin;  // used for polar scattering angle calculation
	double costh = h1*(ese1 - RM)/(ese1 + RM); // polar scattering angle squared
	double sinthe = sqrt(1.0 - costh);
	double costhe = sqrt(costh);

    struct Uphi uphi;
	uphi21(&uphi, costhe, sinthe);

	/* Related change and setup for "new" electron */
	np += 1;
	stack.np = np; // it is needed to update stack index for uphi32()
	stack.iq[np] = -1;
	costh = h1*(ese2 - RM)/(ese2 + RM);
	sinthe = -sqrt(1.0 - costh);
	costhe = sqrt(costh);
	uphi32(&uphi, costhe, sinthe);

    return;
}

void bhabha() {

    /* A call to this function is defined and calculated as a Bhabha scattering
	which impart to the secondary electron enough energy that it will be 
	transported discretely. I.e. E=AE or T=TE. However, it is not guaranteed
	that the final positron will have this much energy. The exact Bhabha 
	differential cross section is used */

	int np = stack.np;
    int irl = stack.ir[np];
    int imed = region.med[irl];
	double eip = stack.e[np];   // total energy of incident positron
	double ekin = eip - RM;     // kinetic energy of incident positron		
	double t0 = ekin/RM;        // kinetic energy of incident positron RM units
	double e0 = t0 + 1.0;       // total energy of incident positron in RM units

	double yy = 1.0/(t0 + 2.0);
	double beta2 = (pow(e0, 2.0) - 1.0)/pow(e0, 2.0);   // incident positron 
													    // velocity in c units
	double ep0 = pegs_data.te[imed]/ekin;   // minimum fractional energy of a 
										    // secondary 'electron'
	double ep0c = 1.0 - ep0;
	double yp = 1.0 - 2.0*yy;

    /* Used in rejection function calculation */
	double b1; double b2; double b3; double b4; 
	b4 = pow(yp, 3.0);
	b3 = b4 + pow(yp, 2.0);
	b2 = yp*(3.0 + pow(yy, 2.0));
	b1 = 2.0 - pow(yy, 2.0);

	/* Sample br from min(ep0) to 1.0 */
	double rnno03; double rnno04;   // random numbers
	double br;      // kinetic energy fraction of the secondary 'electron'
	double rejf2;   // rejection function

	do { 
		rnno03 = setRandom();
		br = ep0/(1.0 - ep0c*rnno03);

		/* Apply rejection function */
		rnno04 = setRandom();
		rejf2 = (1.0 - beta2*br*(b1 - br*(b2 - br*(b3 - br*b4)))); 
	} while(rnno04 > rejf2);

	/* If electron got more than positron, move positron pointer and 
    reflect br */
	if(br < 0.5) { 
		stack.iq[np+1] = -1;
	}
	else { 
		stack.iq[np] = -1;
		stack.iq[np+1] = 1;
		br = 1.0 - br;
		/* This puts positron on top of the stack if it has less energy */
	}

	/* Divide up the energy. */
	br = fmax(br, 0.0);     // avoids possible negative number due to round-off
	double ekse2 = br*ekin;      // kinetic energy of secondary 'electron' 2
	double ese1 = eip - ekse2;   // energy of secondary 'electron' 1
	double ese2 = ekse2 + RM;    // energy of secondary 'electron' 2
	stack.e[np] = ese1;
	stack.e[np+1] = ese2;

	/* Bhabha angles are uniquely determined by kinematics */
	double h1 = (eip + RM)/ekin; // used in direction cosine calculations

	/* Direction cosine change for 'old' electron */
	double costh = fmin(1.0, h1*(ese1 - RM)/(ese1 + RM)); 

	double sinthe = sqrt(1.0 - costh);
	double costhe = sqrt(costh);
    struct Uphi uphi;
	uphi21(&uphi, costhe, sinthe);

	np += 1;
	stack.np = np; // it is needed to update stack index for uphi32()

	costh = h1*(ese2 - RM)/(ese2 + RM);
	sinthe = -sqrt(1.0 - costh);
	costhe = sqrt(costh);
	uphi32(&uphi, costhe, sinthe);

    return;
}

void annih() {

    /* Gamma spectrum for two gamma in-flight positron annihilation using 
	scheme based on Heitler's formulae */
	int np = stack.np;
	double avip = stack.e[np] + RM; // available energy of incident positron, 
									// i.e. electron assumed to be at rest.
	double a = avip/RM; // total energy in units of the electron's rest energy
	double g, t, p; // energy, kinetic energy and momentum in units of RM
	g = a - 1.0;
	t = g - 1.0;
	p = sqrt(a*t);

	double pot = p/t;                       // "p over t"
	double ep0 = 1.0/(a+p);                 // minimum fractional energy
	double wsamp = log((1.0 - ep0)/ep0);    // the logarithm is calculated
										    // outside the loop

	double aa = stack.u[np]; // for inline rotations
	double bb = stack.v[np];
    double cc = stack.w[np];
    double sinpsi = pow(aa, 2.0) + pow(bb, 2.0);
	double sindel; double cosdel;   // for inline rotations

	if(sinpsi > 1.0E-20) { 
		sinpsi = sqrt(sinpsi);
		sindel = bb/sinpsi;
		cosdel = aa/sinpsi;
	}

	double ep;  // fractional energy of the more energetic photon
	double rejf;                    // rejection function
	double rnno01; double rnno02;   // random numbers 

	do { 
		rnno01 = setRandom();
		ep = ep0*exp(rnno01*wsamp);

		/* Now decide whether to accept */
        rnno02 = setRandom();
		rejf = 1.0 - pow(ep*a - 1.0, 2.0)/(ep*(pow(a, 2.0) - 2.0));
	} while(rnno02 > rejf);

	/* Set-up energies. */
	double esg1 = avip*ep;   // energy of secondary gamma 1
	stack.e[np] = esg1;
	stack.iq[np] = 0;

	double costhe = fmin(1.0, (esg1 - RM)*pot/esg1);
	double sinthe = sqrt(1.0 - pow(costhe, 2.0));

	/* The following variables are for azimuthal angle sampling */
	double sphi; double cphi;   // sine and cosine of the azimuthal angle
	double phi = 2.0*M_PI*setRandom();
	sphi = sin(phi);
    cphi = cos(phi);

	double us; double vs; // for inline rotations
	if(sinpsi >= 1.0E-10) { 
		us = sinthe*cphi;
		vs = sinthe*sphi;
		
		stack.u[np] = cc*cosdel*us - sindel*vs + aa*costhe;
        stack.v[np] = cc*sindel*us + cosdel*vs + bb*costhe; 
	    stack.w[np] = cc*costhe - sinpsi*us;
	}
	else { 
        stack.u[np] = sinthe*cphi;
        stack.v[np] = sinthe*sphi; 
	    stack.w[np] = cc*costhe;
	}

	np += 1;
	double esg2 = avip - esg1;
	stack.e[np] = esg2;
	stack.iq[np] = 0;
	transferProperties(np, np-1);

	costhe = fmin(1.0, (esg2 - RM)*pot/esg2);
	sinthe = -sqrt(1.0 - pow(costhe, 2.0));

	if(sinpsi >= 1.0E-10) { 
		us = sinthe*cphi;
		vs = sinthe*sphi;
		
        stack.u[np] = cc*cosdel*us - sindel*vs + aa*costhe;
        stack.v[np] = cc*sindel*us + cosdel*vs + bb*costhe; 
	    stack.w[np] = cc*costhe - sinpsi*us;
	}
	else { 
        stack.u[np] = sinthe*cphi;
        stack.v[np] = sinthe*sphi; 
	    stack.w[np] = cc*costhe;
	}

	/* Update stack index */
	stack.np = np;

    return;
}

int pwlfInterval(int idx, double lvar, double *coef1, double *coef0) {
    
    return (int)(lvar*coef1[idx] + coef0[idx]);
}

double pwlfEval(int idx, double lvar, double *coef1, double *coef0) {
    
    return lvar*coef1[idx] + coef0[idx];
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
