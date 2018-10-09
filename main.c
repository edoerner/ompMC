//
//  main.c
//  ompMC
//
//  Created by Edgardo Dörner on 9/22/18.
//  Copyright © 2018 ED. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/******************************************************************************/
/* Parsing program options with getopt long
 http://www.gnu.org/software/libc/manual/html_node/Getopt.html#Getopt */

#include <getopt.h>

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
void removeSpaces(char* restrict str_trimmed,
                  const char* restrict str_untrimmed);

struct inputItems {
    char key[60];
    char value[60];
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
struct Photon photon;

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

/******************************************************************************/
/* Rayleigh data definition */
#define MXRAYFF 100     // Rayleigh atomic form factor
#define RAYCDFSIZE 100  // CDF from Rayleigh from factors squared

struct Rayleigh {
    double *ff;
    double *xgrid;
    double *fcum;
    double *b_array;
    double *c_array;
    double *pe_array;
    double *pmax0;
    double *pmax1;
    int *i_array;
};

struct Rayleigh rayleigh;

void initRayleighData(void);
void readFfData(double *xval, double **aff);
void cleanRayleigh(void);

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

struct Pair pair;

void initPairData(void);
double fcoulc(double zi);
double xsif(double zi, double fc);
void cleanPair(void);

/******************************************************************************/
/* Electron data definition */

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
    
    double *range_eq;
    double *e_array;
    
    double *eke0;
    double *eke1;
    
    int *sig_ismonotone;
    
    double *esig_e;
    double *psig_e;
    double *xcc;
    double *blcc;
    double *expeke1;
    
    int *iunrst;
    int *epstfl;
    int *iaprim;
    
};

struct Electron electron;

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

struct Mscat mscat;

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

struct Spin spin;

void initElectronData(void);
void cleanElectron(void);
void readRutherfordMscat(int nmed);
void cleanMscat(void);
void initSpinData(int nmed);
void cleanSpin(void);
void setSpline(double *x, double *f, double *a, double *b, double *c,
                double *d,int n);
double spline(double s, double *x, double *a, double *b, double *c,
              double *d, int n);

int main (int argc, char **argv) {
    
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
                input_file = optarg;
                printf ("option -i with value `%s'\n", input_file);
                break;
                
            case 'o':
                output_file = optarg;
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
    
    if(verbose_flag) {
        for (int i = 0; i<input_idx; i++) {
            printf("key = %s, value = %s\n", input_items[i].key,
                   input_items[i].value);
        }
    }
    
    /* Read geometry information from phantom file and initialize geometry */
    initPhantom();
    
    /* With number of media and media names initialize the medium data */
    initMediaData();
    
    /* Cleaning */
    cleanPhantom();
    cleanPhoton();
    cleanRayleigh();
    cleanPair();
    cleanElectron();
    cleanMscat();
    cleanSpin();
    
    exit (EXIT_SUCCESS);
}

void parseInputFile(char *file_name) {
    
    char buf[120];      // support lines up to 120 characters
    
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

void removeSpaces(char* restrict str_trimmed,
                  const char* restrict str_untrimmed) {
    
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
    geometry.xbounds = malloc(geometry.isize*sizeof(double));
    geometry.ybounds = malloc(geometry.jsize*sizeof(double));
    geometry.zbounds = malloc(geometry.ksize*sizeof(double));
    
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
    
    /* Get phantom file path from input file */
    free(geometry.xbounds);
    free(geometry.ybounds);
    free(geometry.zbounds);
    free(geometry.med_indices);
    free(geometry.med_densities);
    
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
    
    /* Initialize data needed for electron interactions */
    initElectronData();
    
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
     Alocate Electron struct arrays */
    electron.blcc = malloc(geometry.nmed*sizeof(double));
    electron.xcc = malloc(geometry.nmed*sizeof(double));
    electron.eke0 = malloc(geometry.nmed*sizeof(double));
    electron.eke1 = malloc(geometry.nmed*sizeof(double));
    electron.esig0 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.esig1 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.psig0 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.psig1 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.ededx0 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.ededx1 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.pdedx0 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.pdedx1 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.ebr10 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.ebr11 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.pbr10 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.pbr11 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.pbr20 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.pbr21 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.tmxs0 = malloc(geometry.nmed*MXEKE*sizeof(double));
    electron.tmxs1 = malloc(geometry.nmed*MXEKE*sizeof(double));
    
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
            int nmedium = 0; // medium index
            
            for (int i = 0; i < geometry.nmed; i++) {
                char cname[20];
                strncpy(cname, geometry.med_names[i], 20);
                if (strcmp(name, cname) == 0) {
                    required = 1;
                    nmedium = i;
                    break;
                }
            }
            if (required == 0) { // return to beginning of the do loop
                continue;
            }
            
            /* We have found the i'th required medium */
            strncpy(pegs_data.names[nmedium], name, 60);
            pegs_data.ne[nmedium] = 0;
            
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
                    pegs_data.rho[nmedium] = d;
                }
                else if (strcmp(name3, "NE") == 0) {
                    int u;
                    if (sscanf(value, "%u", &u) != 1) {
                        ok = 0;
                        break;
                    }
                    pegs_data.ne[nmedium] = u;
                }
                else if (strcmp(name3, "IUNRST") == 0) {
                    int i;
                    if (sscanf(value, "%d", &i) != 1) {
                        ok = 0;
                        break;
                    }
                    pegs_data.iunrst[nmedium] = i;
                }
                else if (strcmp(name3, "EPSTFL") == 0) {
                    int i;
                    if (sscanf(value, "%d", &i) != 1) {
                        ok = 0;
                        break;
                    }
                    pegs_data.epstfl[nmedium] = i;
                }
                else if (strcmp(name3, "IAPRIM") == 0) {
                    int i;
                    if (sscanf(value, "%d", &i) != 1) {
                        ok = 0;
                        break;
                    }
                    pegs_data.iaprim[nmedium] = i;
                }
                token = strtok_r(NULL, ",", &temp);
                i++;
            }
            if (!ok) {
                continue;
            } // end of 2nd line readings
            
            /* Read elements, same algorithm */
            for (int m = 0; m < pegs_data.ne[nmedium]; m++) {
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
                
                pegs_data.elements[nmedium][m] = element;
            }
            
            if (ok == 0) {
                continue;
            }
            
            /* Read next line that contines rlc, ae, ap, ue, up */
            fgets(buffer, 80, fp);
            
            /* The format specifier '%lf' is needed to correctly recognize
             engineering notation. I do not now if this is a property of
             clang, because I had not to do that before */
            if (sscanf(buffer, "%lf %lf %lf %lf %lf\n", &pegs_data.rlc[nmedium],
                       &pegs_data.ae[nmedium], &pegs_data.ap[nmedium],
                       &pegs_data.ue[nmedium], &pegs_data.up[nmedium]) != 5) {
                continue;
            }
            pegs_data.te[nmedium] = pegs_data.ae[nmedium] - RM;
            pegs_data.thmoll[nmedium] = (pegs_data.te[nmedium]) * 2 + RM;
            
            /* Save the medium and mark it found */
            fgets(buffer, 80, fp);
            if (sscanf(buffer, "%d %d %d %d %d %d %d\n",
                       &pegs_data.msge[nmedium], &pegs_data.mge[nmedium],
                       &pegs_data.mseke[nmedium], &pegs_data.meke[nmedium],
                       &pegs_data.mleke[nmedium], &pegs_data.mcmfp[nmedium],
                       &pegs_data.mrange[nmedium]) != 7) {
                continue;
            }
            if (pegs_data.meke[nmedium]>MXEKE) {
                printf("Medium %d has MEKE too big, change MXEKE to %d in "
                       "source code", nmedium, pegs_data.meke[nmedium]);
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
            fscanf(fp, "%lf %lf %lf %lf %lf ", &pegs_data.delcm[nmedium], &ALPHI1,
                   &ALPHI2, &BPAR1, &BPAR2);
            fscanf(fp, "%lf %lf ", &DELPOS1, &DELPOS2);
            fscanf(fp, "%lf %lf %lf %lf ", &XR0, &TEFF0, &electron.blcc[nmedium],
                   &electron.xcc[nmedium]);
            fscanf(fp, "%lf %lf ", &electron.eke0[nmedium],
                   &electron.eke1[nmedium]);
            
            int neke = pegs_data.meke[nmedium];
            for (int k = 0; k<neke; k++) {
                fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf ",
                       &electron.esig0[nmedium*MXEKE + k],
                       &electron.esig1[nmedium*MXEKE + k],
                       &electron.psig0[nmedium*MXEKE + k],
                       &electron.psig1[nmedium*MXEKE + k],
                       &electron.ededx0[nmedium*MXEKE + k],
                       &electron.ededx1[nmedium*MXEKE + k],
                       &electron.pdedx0[nmedium*MXEKE + k],
                       &electron.pdedx1[nmedium*MXEKE + k]);
                
                fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf ",
                       &electron.ebr10[nmedium*MXEKE + k],
                       &electron.ebr11[nmedium*MXEKE + k],
                       &electron.pbr10[nmedium*MXEKE + k],
                       &electron.pbr11[nmedium*MXEKE + k],
                       &electron.pbr20[nmedium*MXEKE + k],
                       &electron.pbr21[nmedium*MXEKE + k],
                       &electron.tmxs0[nmedium*MXEKE + k],
                       &electron.tmxs1[nmedium*MXEKE + k]);
            }
            
            /* length units, only for cm */
            double DFACTI = 1.0 / (pegs_data.rlc[nmedium]);
            electron.blcc[nmedium] *= DFACTI;
            for (int k = 0; k<neke; k++) {
                electron.esig0[nmedium*MXEKE + k] *= DFACTI;
                electron.psig0[nmedium*MXEKE + k] *= DFACTI;
                electron.ededx0[nmedium*MXEKE + k] *= DFACTI;
                electron.pdedx0[nmedium*MXEKE + k] *= DFACTI;
                electron.pdedx1[nmedium*MXEKE + k] *= DFACTI;
                electron.esig1[nmedium*MXEKE + k] *= DFACTI;
                electron.psig1[nmedium*MXEKE + k] *= DFACTI;
                electron.ededx1[nmedium*MXEKE + k] *= DFACTI;
            }
            electron.xcc[nmedium] *= sqrt(DFACTI);
            
            /* Mark the medium found */
            media_found[nmedium] = 1;
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
            printf("\t blcc = %f\n", electron.blcc[i]);
            printf("\t xcc = %f\n", electron.xcc[i]);
            printf("\t eke0 = %f\n", electron.eke0[i]);
            printf("\t eke1 = %f\n", electron.eke1[i]);
            
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
    
    if (getInputValue(buffer, "photon xsection") != 1) {
        printf("Can not find 'photon xsection' key on input file.\n");
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
    strcat(xsection_file, "_photo.data");
    readXsecData(xsection_file, photo_ndat, photo_xsec_data0, photo_xsec_data1);
    
    int *rayleigh_ndat = (int*) malloc(MXELEMENT*sizeof(int));
    double **rayleigh_xsec_data0 = (double**) malloc(MXELEMENT*sizeof(double*));
    double **rayleigh_xsec_data1 = (double**) malloc(MXELEMENT*sizeof(double*));
    
    strcpy(xsection_file, photon_xsection);
    strcat(xsection_file, "_rayleigh.data");
    readXsecData(xsection_file, rayleigh_ndat, rayleigh_xsec_data0,
                 rayleigh_xsec_data1);
    
    int *pair_ndat = (int*) malloc(MXELEMENT*sizeof(int));
    double **pair_xsec_data0 = (double**) malloc(MXELEMENT*sizeof(double*));
    double **pair_xsec_data1 = (double**) malloc(MXELEMENT*sizeof(double*));
    
    strcpy(xsection_file, photon_xsection);
    strcat(xsection_file, "_pair.data");
    readXsecData(xsection_file, pair_ndat, pair_xsec_data0, pair_xsec_data1);
    
    /* We do not consider bound compton scattering, therefore there is no
     cross sections needed for compton scattering */
    
    int *triplet_ndat = (int*) malloc(MXELEMENT*sizeof(int));
    double **triplet_xsec_data0 = (double**) malloc(MXELEMENT*sizeof(double*));
    double **triplet_xsec_data1 = (double**) malloc(MXELEMENT*sizeof(double*));
    
    strcpy(xsection_file, photon_xsection);
    strcat(xsection_file, "_triplet.data");
    readXsecData(xsection_file, triplet_ndat, triplet_xsec_data0,
                 triplet_xsec_data1);
    
    /* binding energies per element removed, as it is not currently supported */
    
    photon.ge0 = malloc(geometry.nmed*sizeof(double));
    photon.ge1 = malloc(geometry.nmed*sizeof(double));
    photon.gmfp0 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon.gmfp1 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon.gbr10 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon.gbr11 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon.gbr20 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon.gbr21 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon.cohe0 = malloc(geometry.nmed*MXGE*sizeof(double));
    photon.cohe1 = malloc(geometry.nmed*MXGE*sizeof(double));
    
    for (int i=0; i<geometry.nmed; i++) {
        photon.ge1[i] = (double)(MXGE - 1)/log(pegs_data.up[i]/pegs_data.ap[i]);
        photon.ge0[i] = 1.0 - photon.ge1[i]*log(pegs_data.ap[i]);
        
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
                                     photon.ge0[i], photon.ge1[i]);
        double *sig_rayleigh = get_data(0, pegs_data.ne[i], rayleigh_ndat,
                                        rayleigh_xsec_data0, rayleigh_xsec_data1
                                        , z_sorted, pz_sorted,
                                        photon.ge0[i], photon.ge1[i]);
        double *sig_pair = get_data(1, pegs_data.ne[i], pair_ndat,
                                    pair_xsec_data0, pair_xsec_data1,
                                    z_sorted, pz_sorted,
                                    photon.ge0[i], photon.ge1[i]);
        double *sig_triplet = get_data(2, pegs_data.ne[i], triplet_ndat,
                                       triplet_xsec_data0, triplet_xsec_data0,
                                       z_sorted, pz_sorted,
                                       photon.ge0[i], photon.ge1[i]);
        
        double gle = 0.0, gmfp = 0.0, gbr1 = 0.0, gbr2 = 0.0, cohe = 0.0;
        double gmfp_old = 0.0, gbr1_old = 0.0, gbr2_old = 0.0,
        cohe_old = 0.0;
        
        for (int j=0; j<MXGE; j++) {
            /* Added +1 to j below due to C loop starting at 0 */
            gle = ((double)(j+1) - photon.ge0[i]) / photon.ge1[i];
            double e = exp(gle);
            double sig_kn = sumZ*kn_sigma0(e);
            
            double sig_p = sig_pair[j] + sig_triplet[j];
            double sigma = sig_kn + sig_p + sig_photo[j];
            gmfp = 1.0/(sigma * con2);
            gbr1 = sig_p/sigma;
            gbr2 = gbr1 + sig_kn/sigma;
            cohe = sigma/(sig_rayleigh[j] + sigma);
            
            if (j > 0) {
                int idx = i*MXGE + (j-1); // the -1 is not for C indexing!
                photon.gmfp1[idx] = (gmfp - gmfp_old)*photon.ge1[i];
                photon.gmfp0[idx] = gmfp - photon.gmfp1[idx]*gle;
                
                photon.gbr11[idx] = (gbr1 - gbr1_old)*photon.ge1[i];
                photon.gbr10[idx] = gbr1 - photon.gbr11[idx]*gle;
                
                photon.gbr21[idx] = (gbr2 - gbr2_old)*photon.ge1[i];
                photon.gbr20[idx] = gbr2 - photon.gbr21[idx]*gle;
                
                photon.cohe1[idx] = (cohe - cohe_old)*photon.ge1[i];
                photon.cohe0[idx] = cohe - photon.cohe1[idx]*gle;
            }
            
            gmfp_old = gmfp;
            gbr1_old = gbr1;
            gbr2_old = gbr2;
            cohe_old = cohe;
        }
        
        int idx = i*MXGE + MXGE - 1; // C-indexing
        photon.gmfp1[idx] = photon.gmfp1[idx-1];
        photon.gmfp0[idx] = gmfp - photon.gmfp1[idx]*gle;
        
        photon.gbr11[idx] = photon.gbr11[idx-1];
        photon.gbr10[idx] = gbr1 - photon.gbr11[idx]*gle;
        
        photon.gbr21[idx] = photon.gbr21[idx-1];
        photon.gbr20[idx] = gbr2 - photon.gbr21[idx]*gle;
        
        photon.cohe1[idx] = photon.cohe1[idx-1];
        photon.cohe0[idx] = cohe - photon.cohe1[idx]*gle;
        
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
    
    /* Print information for debugging purposes*/
    if(verbose_flag) {
        printf("Listing photon data: \n");
        for (int i=0; i<geometry.nmed; i++) {
            printf("For medium %s: \n", geometry.med_names[i]);
            printf("\t ge0 = %f, ge1 = %f\n", photon.ge0[i], photon.ge1[i]);
            
            printf("gmfp = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXGE + j;
                printf("gmfp0 = %f, gmfp1 = %f\n", photon.gmfp0[idx],
                       photon.gmfp1[idx]);
            }
            printf("\n");
            
            printf("gbr1 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXGE + j;
                printf("gbr10 = %f, gbr11 = %f\n", photon.gbr10[idx],
                       photon.gbr11[idx]);
            }
            printf("\n");
            
            printf("gbr2 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXGE + j;
                printf("gbr20 = %f, gbr21 = %f\n", photon.gbr20[idx],
                       photon.gbr21[idx]);
            }
            printf("\n");
            
            printf("cohe = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXGE + j;
                printf("cohe0 = %f, cohe1 = %f\n", photon.cohe0[idx],
                       photon.cohe1[idx]);
            }
            printf("\n");
            
        }
    }
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
    
    free(photon.ge0);
    free(photon.ge1);
    free(photon.gmfp0);
    free(photon.gmfp1);
    free(photon.gbr10);
    free(photon.gbr11);
    free(photon.gbr20);
    free(photon.cohe0);
    free(photon.cohe1);
    
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
    
    for (int i=0; i<MXELEMENT; i++) {
        aff[i] = malloc(MXRAYFF*sizeof(double));
    }
    
    /* Read Rayleigh atomic form factor from pgs4form.dat file */
    readFfData(xval, aff);
    
    /* Allocate memory for Rayleigh data */
    rayleigh.ff = malloc(MXRAYFF*geometry.nmed*sizeof(double));
    rayleigh.xgrid = malloc(MXRAYFF*geometry.nmed*sizeof(double));
    rayleigh.fcum = malloc(MXRAYFF*geometry.nmed*sizeof(double));
    rayleigh.b_array = malloc(MXRAYFF*geometry.nmed*sizeof(double));
    rayleigh.c_array = malloc(MXRAYFF*geometry.nmed*sizeof(double));
    rayleigh.i_array = malloc(RAYCDFSIZE*geometry.nmed*sizeof(int));
    rayleigh.pe_array = malloc(MXGE*geometry.nmed*sizeof(double));
    rayleigh.pmax0 = malloc(MXGE*geometry.nmed*sizeof(double));
    rayleigh.pmax1 = malloc(MXGE*geometry.nmed*sizeof(double));
    
    for (int i=0; i<geometry.nmed; i++) {
        /* Calculate form factor using independent atom model */
        for (int j=0; j<MXRAYFF; j++) {
            double ff_val = 0.0;
            rayleigh.xgrid[i*MXRAYFF + j] = xval[j];
            
            for (int k=0; k<pegs_data.ne[i]; k++) {
                int z = (int)pegs_data.elements[i][k].z - 1; // C indexing
                ff_val += pegs_data.elements[i][k].pz * pow(aff[z][j],2);
            }
            
            rayleigh.ff[i*MXRAYFF + j] = sqrt(ff_val);
        }
        
        if (rayleigh.xgrid[i*MXRAYFF] < 1.0E-6) {
            rayleigh.xgrid[i*MXRAYFF] = 0.0001;
        }
        
        /* Calculate rayleigh data, as in subroutine prepare_rayleigh_data
         inside EGSnrc*/
        double emin = exp((1.0 - photon.ge0[i])/photon.ge1[i]);
        double emax = exp((MXGE - photon.ge0[i])/photon.ge1[i]);
        
        /* The following is to avoid log(0) */
        for (int j=0; j<MXRAYFF; j++) {
            if (*((unsigned long*)&rayleigh.ff[i*MXRAYFF + j]) == 0) {
                unsigned long zero = 1;
                rayleigh.ff[i*MXRAYFF + j] = *((double*)&zero);
            }
        }
        
        /* Calculating the cumulative distribution */
        double sum0 = 0.0;
        rayleigh.fcum[i*MXRAYFF] = 0.0;
        
        for (int j=0; j < MXRAYFF-1; j++) {
            double b = log(rayleigh.ff[i*MXRAYFF + j + 1]
                           /rayleigh.ff[i*MXRAYFF + j])
                                /log(rayleigh.xgrid[i*MXRAYFF + j + 1]
                                /rayleigh.xgrid[i*MXRAYFF + j]);
            rayleigh.b_array[i*MXRAYFF + j] = b;
            double x1 = rayleigh.xgrid[i*MXRAYFF + j];
            double x2 = rayleigh.xgrid[i*MXRAYFF + j + 1];
            double pow_x1 = pow(x1, 2.0*b);
            double pow_x2 = pow(x2, 2.0*b);
            sum0 += pow(rayleigh.ff[i*MXRAYFF + j],2)
                *(pow(x2,2)*pow_x2 - pow(x1,2)*pow_x1)/((1.0 + b)*pow_x1);
            rayleigh.fcum[i*MXRAYFF + j + 1] = sum0;
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
                if ((xmax >= rayleigh.xgrid[i*MXRAYFF + k - 1]) &&
                    (xmax < rayleigh.xgrid[i*MXRAYFF + k]))
                    break;
            }
            
            idx = k;
            double b = rayleigh.b_array[i*MXRAYFF + idx - 1];
            double x1 = rayleigh.xgrid[i*MXRAYFF + idx - 1];
            double x2 = xmax;
            double pow_x1 = pow(x1, 2.0 * b);
            double pow_x2 = pow(x2, 2.0 * b);
            rayleigh.pe_array[i*MXGE + j - 1] =
                rayleigh.fcum[i*MXRAYFF + idx - 1] +
                pow(rayleigh.ff[i * MXRAYFF + idx - 1], 2) *
                (pow(x2,2)*pow_x2 - pow(x1,2)*pow_x1)/((1.0 + b)*pow_x1);
            
        }
        
        rayleigh.i_array[i*RAYCDFSIZE + RAYCDFSIZE - 1] = idx;
        
        /* Now renormalize data so that pe_array(emax) = 1. Note that we make
         pe_array(j) slightly larger so that fcum(xmax) is never underestimated
         when interpolating */
        double anorm = 1.0/sqrt(rayleigh.pe_array[i*MXGE + MXGE - 1]);
        double anorm1 = 1.005/rayleigh.pe_array[i*MXGE + MXGE - 1];
        double anorm2 = 1.0/rayleigh.pe_array[i*MXGE + MXGE - 1];
        
        for (int j=0; j<MXGE; j++) {
            rayleigh.pe_array[i*MXGE + j] *= anorm1;
            if (rayleigh.pe_array[i*MXGE + j] > 1.0) {
                rayleigh.pe_array[i*MXGE + j] = 1.0;
            }
        }
        
        for (int j=0; j<MXRAYFF; j++) {
            rayleigh.ff[i*MXRAYFF + j] *= anorm;
            rayleigh.fcum[i*MXRAYFF + j] *= anorm2;
            rayleigh.c_array[i*MXRAYFF + j] = (1.0 +
                rayleigh.b_array[i*MXRAYFF + j])/
            pow(rayleigh.xgrid[i*MXRAYFF + j]*rayleigh.ff[i*MXRAYFF + j],2);
        }
        
        /* Now prepare uniform cumulative bins */
        double dw = 1.0/((double)RAYCDFSIZE - 1.0);
        double xold = rayleigh.xgrid[i*MXRAYFF + 0];
        int ibin = 1;
        double b = rayleigh.b_array[i*MXRAYFF + 0];
        double pow_x1 = pow(rayleigh.xgrid[i*MXRAYFF + 0], 2.0*b);
        rayleigh.i_array[i*MXRAYFF + 0] = 1;
        
        for (int j=2; j<=RAYCDFSIZE-1; j++) {
            double w = dw;
            
            do {
                double x1 = xold;
                double x2 = rayleigh.xgrid[i*MXRAYFF + ibin];
                double t = pow(x1, 2)*pow(x1, 2.0*b);
                double pow_x2 = pow(x2, 2.0*b);
                double aux = pow(rayleigh.ff[i*MXRAYFF + ibin - 1], 2)*
                    (pow(x2, 2)*pow_x2 - t)/((1.0 + b)*pow_x1);
                if (aux > w) {
                    xold = exp(log(t + w*(1.0 + b)*pow_x1/
                                   pow(rayleigh.ff[i*MXRAYFF + ibin - 1], 2))/
                               (2.0 + 2.0*b));
                    rayleigh.i_array[i*RAYCDFSIZE + j - 1] = ibin;
                    break;
                }
                w -= aux;
                xold = x2;
                ibin++;
                b = rayleigh.b_array[i*MXRAYFF + ibin - 1];
                pow_x1 = pow(xold, 2.0*b);
            } while (1);
        }
        
        /* change definition of b_array because that is what is needed at
         run time*/
        for (int j=0; j<MXRAYFF; j++) {
            rayleigh.b_array[i*MXRAYFF + j] = 0.5/(1.0 +
                rayleigh.b_array[i*MXRAYFF + j]);
        }
        
        /* Prepare coefficients for pmax interpolation */
        for (int j=0; j<MXGE-1; j++) {
            double gle = ((j+1) - photon.ge0[i])/photon.ge1[i];
            rayleigh.pmax1[i*MXGE + j] = (rayleigh.pe_array[i*MXGE + j + 1] -
                rayleigh.pe_array[i * MXGE + j])*photon.ge1[i];
            rayleigh.pmax0[i*MXGE + j] = rayleigh.pe_array[i*MXGE + j] -
                rayleigh.pmax1[i * MXGE + j]*gle;
        }
        rayleigh.pmax0[i*MXGE + MXGE - 1] = rayleigh.pmax0[i*MXGE + MXGE - 2];
        rayleigh.pmax1[i*MXGE + MXGE - 1] = rayleigh.pmax1[i*MXGE + MXGE - 2];
    }
    
    /* Cleaning */
    free(xval);
    for (int i=0; i<MXELEMENT; i++) {
        free(aff[i]);
    }
    free(aff);
    
    /* Print information for debugging purposes*/
    if(verbose_flag) {
        printf("Listing rayleigh data: \n");
        for (int i=0; i<geometry.nmed; i++) {
            printf("For medium %s: \n", geometry.med_names[i]);
            
            printf("ff = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXRAYFF + j;
                printf("ff = %f\n", rayleigh.ff[idx]);
            }
            printf("\n");
            
            printf("xgrid = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXRAYFF + j;
                printf("xgrid = %f\n", rayleigh.xgrid[idx]);
            }
            printf("\n");
            
            printf("fcum = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXRAYFF + j;
                printf("fcum = %f\n", rayleigh.fcum[idx]);
            }
            printf("\n");
            
            printf("b_array = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXRAYFF + j;
                printf("b_array = %f\n", rayleigh.b_array[idx]);
            }
            printf("\n");
            
            printf("c_array = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXRAYFF + j;
                printf("c_array = %f\n", rayleigh.c_array[idx]);
            }
            printf("\n");
            
            printf("pe_array = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXGE + j;
                printf("pe_array = %f\n", rayleigh.pe_array[idx]);
            }
            printf("\n");
            
            printf("pmax = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*MXGE + j;
                printf("pmax0 = %f, pmax1 = %f\n", rayleigh.pmax0[idx],
                       rayleigh.pmax1[idx]);
            }
            printf("\n");
            
            printf("i_array = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*RAYCDFSIZE + j;
                printf("i_array = %d\n", rayleigh.i_array[idx]);
            }
            printf("\n");
            
        }
    }
    
    return;
}

void readFfData(double *xval, double **aff) {
    
    FILE *fp;
    char file_name[25] = "./pegs4/pgs4form.dat";
    
    /* Open file containing form factors */
    if ((fp = fopen(file_name, "r")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
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
        printf("Could not read atomic form factors file %s", file_name);
        exit(EXIT_FAILURE);
    }
    
    return;
}

void cleanRayleigh() {
    
    free(rayleigh.b_array);
    free(rayleigh.c_array);
    free(rayleigh.fcum);
    free(rayleigh.ff);
    free(rayleigh.i_array);
    free(rayleigh.pe_array);
    free(rayleigh.pmax0);
    free(rayleigh.pmax1);
    
    return;
}

void initPairData() {
    /* The data calculated here corresponds partially to the subroutine
     fix_brems in egsnrc.macros. This subroutine calculates the parameters
     for the rejection function used in bremsstrahlung sampling*/

    double Zf, Zb, Zt, Zg, Zv; // medium functions, used for delx and delcm
    double fmax1, fmax2;
    
    /* Memory allocation */
    pair.dl1 = malloc(geometry.nmed*8*sizeof(double));
    pair.dl2 = malloc(geometry.nmed*8*sizeof(double));
    pair.dl3 = malloc(geometry.nmed*8*sizeof(double));
    pair.dl4 = malloc(geometry.nmed*8*sizeof(double));
    pair.dl5 = malloc(geometry.nmed*8*sizeof(double));
    pair.dl6 = malloc(geometry.nmed*8*sizeof(double));
    
    pair.bpar0 = malloc(geometry.nmed*sizeof(double));
    pair.bpar1 = malloc(geometry.nmed*sizeof(double));
    pair.delcm = malloc(geometry.nmed*sizeof(double));
    pair.zbrang = malloc(geometry.nmed*sizeof(double));
    
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
        pair.dl1[imed*8 + 0] = (20.863 + 4.0*Zg)/fmax1;
        pair.dl2[imed*8 + 0] = -3.242/fmax1;
        pair.dl3[imed*8 + 0] = 0.625/fmax1;
        pair.dl4[imed*8 + 0] = (21.12 + 4.0*Zg)/fmax1;
        pair.dl5[imed*8 + 0] = -4.184/fmax1;
        pair.dl6[imed*8 + 0] = 0.952;
        
        pair.dl1[imed*8 + 1] = (20.029 + 4.0*Zg)/fmax1;
        pair.dl2[imed*8 + 1] = -1.93/fmax1;
        pair.dl3[imed*8 + 1] = -0.086/fmax1;
        pair.dl4[imed*8 + 1] = (21.12 + 4.0*Zg)/fmax1;
        pair.dl5[imed*8 + 1] = -4.184/fmax1;
        pair.dl6[imed*8 + 1] = 0.952;
        
        pair.dl1[imed*8 + 2] = (20.863 + 4.0*Zv)/fmax2;
        pair.dl2[imed*8 + 2] = -3.242/fmax2;
        pair.dl3[imed*8 + 2] = 0.625/fmax2;
        pair.dl4[imed*8 + 2] = (21.12 + 4.0*Zv)/fmax2;
        pair.dl5[imed*8 + 2] = -4.184/fmax2;
        pair.dl6[imed*8 + 2] = 0.952;
        
        pair.dl1[imed*8 + 3] = (20.029 + 4.0*Zv)/fmax2;
        pair.dl2[imed*8 + 3] = -1.93/fmax2;
        pair.dl3[imed*8 + 3] = -0.086/fmax2;
        pair.dl4[imed*8 + 3] = (21.12 + 4.0*Zv)/fmax2;
        pair.dl5[imed*8 + 3] = -4.184/fmax2;
        pair.dl6[imed*8 + 3] = 0.952;
        
        // The following data are used in pair production.
        pair.dl1[imed*8 + 4] = (3.0*(20.863 + 4.0*Zg) - (20.029 + 4.0*Zg));
        pair.dl2[imed*8 + 4] = (3.0*(-3.242) - (-1.930));
        pair.dl3[imed*8 + 4] = (3.0*(0.625) - (-0.086));
        pair.dl4[imed*8 + 4] = (2.0*21.12 + 8.0*Zg);
        pair.dl5[imed*8 + 4] = (2.0*(-4.184));
        pair.dl6[imed*8 + 4] = 0.952;
        
        pair.dl1[imed*8 + 5] = (3.0*(20.863 + 4.0*Zg) + (20.029 + 4.0*Zg));
        pair.dl2[imed*8 + 5] = (3.0*(-3.242) + (-1.930));
        pair.dl3[imed*8 + 5] = (3.0*0.625 + (-0.086));
        pair.dl4[imed*8 + 5] = (4.0*21.12 + 16.0*Zg);
        pair.dl5[imed*8 + 5] = (4.0*(-4.184));
        pair.dl6[imed*8 + 5] = 0.952;
        
        pair.dl1[imed*8 + 6] = (3.0*(20.863 + 4.0*Zv) - (20.029 + 4.0*Zv));
        pair.dl2[imed*8 + 6] = (3.0*(-3.242) - (-1.930));
        pair.dl3[imed*8 + 6] = (3.0*(0.625) - (-0.086));
        pair.dl4[imed*8 + 6] = (2.0*21.12 + 8.0*Zv);
        pair.dl5[imed*8 + 6] = (2.0*(-4.184));
        pair.dl6[imed*8 + 6] = 0.952;
        
        pair.dl1[imed*8 + 7] = (3.0*(20.863 + 4.0*Zv) + (20.029 + 4.0*Zv));
        pair.dl2[imed*8 + 7] = (3.0*(-3.242) + (-1.930));
        pair.dl3[imed*8 + 7] = (3.0*0.625 + (-0.086));
        pair.dl4[imed*8 + 7] = (4.0*21.12 + 16.0*Zv);
        pair.dl5[imed*8 + 7] = (4.0*(-4.184));
        pair.dl6[imed*8 + 7] = 0.952;
        
        pair.bpar1[imed] = pair.dl1[imed*8 + 6]/
            (3.0*pair.dl1[imed*8 + 7] + pair.dl1[imed*8 + 6]);
        pair.bpar0[imed] = 12.0 * pair.dl1[imed*8 +7]/
            (3.0*pair.dl1[imed*8 + 7] + pair.dl1[imed*8 + 6]);
        
        // The following is the calculation of the composite factor for angular
        // distributions, as carried out in $INITIALIZE-PAIR-ANGLE macro. It
        // corresponds to ( (1/111)*Zeff**(1/3) )**2
        float zbrang = 0.0;
        float pznorm = 0.0;
        
        for (int i = 0; i<pegs_data.ne[imed]; i++) {
            zbrang += (float)
            (pegs_data.elements[imed][i].pz)*
            (pegs_data.elements[imed][i].z)*
            ((pegs_data.elements[imed][i].z) + 1.0f);
            pznorm += pegs_data.elements[imed][i].pz;
        }
        pair.zbrang[imed] = (8.116224E-05)*pow(zbrang/pznorm, 1.0/3.0);
        pair.delcm[imed] = pegs_data.delcm[imed];
    }
    
    /* Print information for debugging purposes */
    if(verbose_flag) {
        printf("Listing pair data: \n");
        for (int i=0; i<geometry.nmed; i++) {
            printf("For medium %s: \n", geometry.med_names[i]);
            
            printf("delcm = %f\n", pair.delcm[i]);
            printf("bpar0 = %f\n", pair.bpar0[i]);
            printf("bpar1 = %f\n", pair.bpar1[i]);
            printf("zbrang = %f\n", pair.zbrang[i]);
            
            printf("dl1 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*8 + j;
                printf("dl1 = %f\n", pair.dl1[idx]);
            }
            printf("\n");
            
            printf("dl2 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*8 + j;
                printf("dl2 = %f\n", pair.dl2[idx]);
            }
            printf("\n");
            
            printf("dl3 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*8 + j;
                printf("dl3 = %f\n", pair.dl3[idx]);
            }
            printf("\n");
            
            printf("dl4 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*8 + j;
                printf("dl4 = %f\n", pair.dl4[idx]);
            }
            printf("\n");
            
            printf("dl5 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*8 + j;
                printf("dl5 = %f\n", pair.dl5[idx]);
            }
            printf("\n");
            
            printf("dl6 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*8 + j;
                printf("dl6 = %f\n", pair.dl6[idx]);
            }
            printf("\n");
            
        }
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
    
    free(pair.dl1);
    free(pair.dl2);
    free(pair.dl3);
    free(pair.dl4);
    free(pair.dl5);
    free(pair.dl6);
    
    free(pair.bpar0);
    free(pair.bpar1);
    free(pair.delcm);
    free(pair.zbrang);
    
    return;
}

void initElectronData() {
    
    double temp, eil = 0, ei = 0, sig, sige_old, sigp_old;
    int leil;
    int ise_monoton = 1, isp_monoton = 1;   // i.e., both true by default.
    
    readRutherfordMscat(geometry.nmed);
    
    int neke;
    
    for (int i=0; i<geometry.nmed; i++) {
        /* Absorb Euler constant into the multiple scattering parameter */
        electron.blcc[i] = 1.16699413758864573*electron.blcc[i];
        
        /* Take its square as this is employed throughout */
        electron.xcc[i] = pow(electron.xcc[i], 2);
    }
    
    /* Initialize data for spin effects */
    initSpinData(geometry.nmed);
    
    return;
}

void cleanElectron() {
    
    free(electron.blcc);
    free(electron.blcce0);
    free(electron.blcce1);
    free(electron.e_array);
    free(electron.ebr10);
    free(electron.ebr11);
    free(electron.ededx0);
    free(electron.ededx1);
    free(electron.eke0);
    free(electron.eke1);
    free(electron.epstfl);
    free(electron.esig0);
    free(electron.esig1);
    free(electron.esig_e);
    free(electron.etae_ms0);
    free(electron.etae_ms1);
    free(electron.etap_ms0);
    free(electron.etap_ms1);
    free(electron.expeke1);
    free(electron.iaprim);
    free(electron.iunrst);
    free(electron.pbr10);
    free(electron.pbr11);
    free(electron.pbr20);
    free(electron.pbr21);
    free(electron.pdedx0);
    free(electron.pdedx1);
    free(electron.psig0);
    free(electron.psig1);
    free(electron.psig_e);
    free(electron.q1ce_ms0);
    free(electron.q1ce_ms1);
    free(electron.q1cp_ms0);
    free(electron.q1cp_ms1);
    free(electron.q2ce_ms0);
    free(electron.q2ce_ms1);
    free(electron.q2cp_ms0);
    free(electron.q2cp_ms1);
    free(electron.range_eq);
    free(electron.tmxs0);
    free(electron.tmxs1);
    free(electron.xcc);
    free(electron.sig_ismonotone);
    
    return;
}

void readRutherfordMscat(int nmed) {
    
    /* Open multi-scattering file */
    FILE *fp;
    char file[25] = "./data/msnew.data";
    
    if ((fp = fopen(file, "r")) == NULL) {
        printf("Unable to open file: %s\n", file);
        exit(EXIT_FAILURE);
    }
    
    printf("Path to multi-scattering data file : %s\n", file);
    
    /* Allocate memory for MS data */
    mscat.ums_array =
        malloc((MXL_MS + 1)*(MXQ_MS + 1)*(MXU_MS + 1)*sizeof(double));
    mscat.fms_array =
        malloc((MXL_MS + 1)*(MXQ_MS + 1)*(MXU_MS + 1)*sizeof(double));
    mscat.wms_array =
        malloc((MXL_MS + 1)*(MXQ_MS + 1)*(MXU_MS + 1)*sizeof(double));
    mscat.ims_array =
        malloc((MXL_MS + 1)*(MXQ_MS + 1)*(MXU_MS + 1)*sizeof(double));
    
    printf("Reading multi-scattering data from file : %s\n", file);
    
    for (int i=0; i<=MXL_MS; i++) {
        for (int j=0; j <= MXQ_MS; j++) {
            int k, idx;
            
            for (k=0; k<=MXU_MS; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fscanf(fp, "%lf ", &mscat.ums_array[idx]);
            }
            for (k = 0; k<=MXU_MS; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fscanf(fp, "%lf ", &mscat.fms_array[idx]);
            }
            for (k = 0; k<=MXU_MS-1; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fscanf(fp, "%lf ", &mscat.wms_array[idx]);
            }
            for (k = 0; k<=MXU_MS-1; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                fscanf(fp, "%d ", &mscat.ims_array[idx]);
            }
            
            for (k=0; k<=MXU_MS-1; k++) {
                idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + k;
                mscat.fms_array[idx] = mscat.fms_array[idx + 1]/mscat.fms_array[idx] - 1.0;
                mscat.ims_array[idx] = mscat.ims_array[idx] - 1;
            }
            idx = i*(MXQ_MS + 1)*(MXU_MS + 1) + j*(MXU_MS + 1) + MXU_MS;
            mscat.fms_array[idx] = mscat.fms_array[idx-1];
        }
    }

    double llammin = log(LAMBMIN_MS);
    double llammax = log(LAMBMAX_MS);
    double dllamb  = (llammax-llammin)/MXL_MS;
    mscat.dllambi = 1.0/dllamb;
    
    double dqms    = QMAX_MS/MXQ_MS;
    mscat.dqmsi = 1.0/dqms;
    
    /* Print information for debugging purposes */
    if(verbose_flag) {
        printf("Listing multi-scattering data: \n");
        printf("dllambi = %f\n", mscat.dllambi);
        printf("dqmsi = %f\n", mscat.dqmsi);
        
        printf("ums_array = \n");
        for (int j=0; j<5; j++) { // print just 5 first values
            printf("ums_array = %f\n", mscat.ums_array[j]);
        }
        printf("\n");
        
        printf("fms_array = \n");
        for (int j=0; j<5; j++) { // print just 5 first values
            printf("fms_array = %f\n", mscat.fms_array[j]);
        }
        printf("\n");
        
        printf("wms_array = \n");
        for (int j=0; j<5; j++) { // print just 5 first values
            printf("wms_array = %f\n", mscat.wms_array[j]);
        }
        printf("\n");
        
        printf("ims_array = \n");
        for (int j=0; j<5; j++) { // print just 5 first values
            printf("ims_array = %d\n", mscat.ims_array[j]);
        }
        printf("\n");
        
    }
    
    fclose(fp);
    
    return;
}

void cleanMscat() {
    
    free(mscat.fms_array);
    free(mscat.ims_array);
    free(mscat.ums_array);
    free(mscat.wms_array);
    
    return;
}

void initSpinData(int nmed) {
    
    /* Open multi-scattering file */
    FILE *fp;
    char file[25] = "./data/spinms.data";
    
    if ((fp = fopen(file, "r")) == NULL) {
        printf("Unable to open file: %s\n", file);
        exit(EXIT_FAILURE);
    }
    
    printf("Path to spin data file : %s\n", file);
    
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
    spin.b2spin_min = (double)b2spin_min;
    
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
    spin.dleneri = 1.0/dlener;
    spin.espml = log(espin_min);
    dbeta2 = (b2spin_max - b2spin_min)/MXE_SPIN;
    spin.dbeta2i = 1.0/dbeta2;
    double dqq1 = 0.5/MXQ_SPIN;
    spin.dqq1i = 1.0/dqq1;
    
    double *eta_array = (double*) malloc(2*(MXE_SPIN1+1)*sizeof(double));
    double *c_array = (double*) malloc(2*(MXE_SPIN1+1)*sizeof(double));
    double *g_array = (double*) malloc(2*(MXE_SPIN1+1)*sizeof(double));
    
    spin.spin_rej = calloc(nmed*2*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)*
                           (MXU_SPIN + 1), sizeof(double));
    
    double *fmax_array = (double*) malloc((MXQ_SPIN + 1)*sizeof(double));
    
    /* Needed for correction to first MS moment due to spin effects */
    double *elarray = (double*) malloc((MXE_SPIN1 + 1)*sizeof(double));
    double *farray = (double*) malloc((MXE_SPIN1 + 1)*sizeof(double));
    
    double *af = (double*) malloc((MXE_SPIN1+1)*sizeof(double));
    double *bf = (double*) malloc((MXE_SPIN1+1)*sizeof(double));
    double *cf = (double*) malloc((MXE_SPIN1+1)*sizeof(double));
    double *df = (double*) malloc((MXE_SPIN1+1)*sizeof(double));
    
    /* Allocate memory for electron data */
    electron.etae_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron.etae_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron.etap_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron.etap_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron.q1ce_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron.q1ce_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron.q1cp_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron.q1cp_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron.q2ce_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron.q2ce_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron.q2cp_ms0 = malloc(nmed*MXEKE*sizeof(double));
    electron.q2cp_ms1 = malloc(nmed*MXEKE*sizeof(double));
    electron.blcce0 = malloc(nmed*MXEKE*sizeof(double));
    electron.blcce1 = malloc(nmed*MXEKE*sizeof(double));
    
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
                            spin.spin_rej[imed*2*(MXE_SPIN1 + 1)*
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
                        if (flmax<spin.spin_rej[imed*2*(MXE_SPIN1 + 1)*
                                           (MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                           + iq*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)
                                           *(MXU_SPIN + 1)
                                           + i*(MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                           + j*(MXU_SPIN + 1) + k]) {
                            flmax = spin.spin_rej[imed*2*(MXE_SPIN1 + 1)*
                                             (MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                             + iq*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)
                                             *(MXU_SPIN + 1)
                                             + i*(MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                             + j*(MXU_SPIN + 1) + k];
                        }
                    }
                    for (int k = 0; k <= MXU_SPIN; k++) {
                        spin.spin_rej[imed*2*(MXE_SPIN1 + 1)*
                                 (MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                 + iq*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)
                                 *(MXU_SPIN + 1)
                                 + i*(MXQ_SPIN + 1)*(MXU_SPIN + 1)
                                 + j*(MXU_SPIN + 1) + k] =
                        spin.spin_rej[imed*2*(MXE_SPIN1 + 1)*
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
                (electron.blcc[imed])/(electron.xcc[imed]);
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
        double eil = (1.0 - electron.eke0[imed])/electron.eke1[imed];
        double e = exp(eil);
        double si1e, si1p, si2e, si2p, aae;
        int je = 0;
        
        if (e<=espin_min) {
            si1e = eta_array[0*(MXE_SPIN1 + 1) + 0];
            si1p = eta_array[1*(MXE_SPIN1 + 1) + 0];
        }
        else {
            if (e <= espin_max) {
                aae = (eil - spin.espml)*spin.dleneri;
                je = (int)aae;
                aae = aae - je;
            }
            else {
                tau = e/RM;
                beta2 = tau*(tau + 2.0)/pow(tau + 1.0, 2.0);
                aae = (beta2 - spin.b2spin_min)*spin.dbeta2i;
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
            eil = (i + 1.0 - electron.eke0[imed])/electron.eke1[imed];
            e = exp(eil);
            if (e<=espin_min) {
                si2e = eta_array[0*(MXE_SPIN1 + 1) + 0];
                si2p = eta_array[1*(MXE_SPIN1 + 1) + 0];
            }
            else {
                if (e<=espin_max) {
                    aae = (eil - spin.espml)*spin.dleneri;
                    je = (int)aae;
                    aae = aae - je;
                }
                else {
                    tau = e/RM;
                    beta2 = tau*(tau + 2.0)/((tau + 1.0)*(tau + 1.0));
                    aae = (beta2 - spin.b2spin_min)*spin.dbeta2i;
                    je = (int)aae;
                    aae = aae - je;
                    je = je + MXE_SPIN + 1;
                }
                si2e = (1.0 - aae)*eta_array[0*(MXE_SPIN1 + 1) + je] +
                    aae*eta_array[0*(MXE_SPIN1 + 1) + (je + 1)];
                si2p = (1.0 - aae)*eta_array[1*(MXE_SPIN1 + 1) + je] +
                    aae*eta_array[1*(MXE_SPIN1 + 1) + (je + 1)];
                
            }
            
            electron.etae_ms1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron.eke1[imed];
            electron.etae_ms0[MXEKE*imed + i - 1] =
                (si2e - electron.etae_ms1[MXEKE*imed + i - 1]*eil);
            electron.etap_ms1[MXEKE*imed + i - 1] =
                (si2p - si1p)*electron.eke1[imed];
            electron.etap_ms0[MXEKE*imed + i - 1] =
                (si2p - electron.etap_ms1[MXEKE*imed + i - 1]*eil);
            si1e = si2e; si1p = si2p;
        }
        
        electron.etae_ms1[MXEKE*imed + neke - 1] =
            electron.etae_ms1[MXEKE*imed + neke - 2];
        electron.etae_ms0[MXEKE*imed + neke - 1] =
            electron.etae_ms0[MXEKE*imed + neke - 2];
        electron.etap_ms1[MXEKE*imed + neke - 1] =
            electron.etap_ms1[MXEKE*imed + neke - 2];
        electron.etap_ms0[MXEKE*imed + neke - 1] =
            electron.etap_ms0[MXEKE*imed + neke - 2];

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
        eil = (1.0 - electron.eke0[imed])/electron.eke1[imed];
        si1e = spline(eil, elarray, af, bf, cf, df, ndata);
        
        for(int i=1; i<=neke - 1; i++){
            eil = (i + 1 - electron.eke0[imed])/electron.eke1[imed];
            si2e = spline(eil, elarray, af, bf, cf, df, ndata);
            electron.q1ce_ms1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron.eke1[imed];
            electron.q1ce_ms0[MXEKE*imed + i - 1]=
                si2e - electron.q1ce_ms1[MXEKE*imed + i - 1]*eil;
            si1e = si2e;
        }
        electron.q1ce_ms1[MXEKE*imed + neke - 1] =
            electron.q1ce_ms1[MXEKE*imed + neke - 2];
        electron.q1ce_ms0[MXEKE*imed + neke - 1] =
            electron.q1ce_ms1[MXEKE*imed + neke - 2];
        
        /* Now positrons */
        for (int i=0; i<=MXE_SPIN; i++){
            farray[i] = c_array[1*(MXE_SPIN1 + 1) + i];
        }
        for (int i=MXE_SPIN+1; i<=MXE_SPIN1 - 1; i++){
            farray[i] = c_array[1*(MXE_SPIN1 + 1) + i + 1];
        }
        
        setSpline(elarray, farray, af, bf, cf, df, ndata);
        eil = (1.0 - electron.eke0[imed])/electron.eke1[imed];
        si1e = spline(eil, elarray, af, bf, cf, df, ndata);
        
        for (int i=1; i<=neke-1; i++){
            eil = (i + 1 - electron.eke0[imed])/electron.eke1[imed];
            si2e = spline(eil, elarray, af, bf, cf, df, ndata);
            electron.q1cp_ms1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron.eke1[imed];
            electron.q1cp_ms0[MXEKE*imed + i - 1]=
                si2e - electron.q1cp_ms1[MXEKE*imed + i - 1]*eil;
            si1e = si2e;
        }
        electron.q1cp_ms1[MXEKE*imed + neke - 1] =
            electron.q1cp_ms1[MXEKE*imed + neke - 2];
        electron.q1cp_ms0[MXEKE*imed + neke - 1] =
            electron.q1cp_ms1[MXEKE*imed + neke - 2];
        
        /* Prepare interpolation table for the second MS moment correction */
        /* First electrons */
        for (int i=0; i<=MXE_SPIN; i++) {
            farray[i] = g_array[0*(MXE_SPIN1+1) + i];
        }
        for (int i=MXE_SPIN + 1; i<=MXE_SPIN1 - 1; i++) {
            farray[i] = g_array[0*(MXE_SPIN1+1) + i + 1];
        }
        
        setSpline(elarray, farray, af, bf, cf, df, ndata);
        eil = (1.0 - electron.eke0[imed])/electron.eke1[imed];
        si1e = spline(eil, elarray, af, bf, cf, df, ndata);
        
        for (int i=1; i<=neke-1; i++){
            eil = (i + 1 - electron.eke0[imed])/electron.eke1[imed];
            si2e = spline(eil, elarray, af, bf, cf, df, ndata);
            electron.q2ce_ms1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron.eke1[imed];
            electron.q2ce_ms0[MXEKE*imed + i - 1] =
                si2e - electron.q2ce_ms1[MXEKE*imed + i - 1]*eil;
            si1e = si2e;
        }
        electron.q2ce_ms1[MXEKE*imed + neke - 1] =
            electron.q2ce_ms1[MXEKE*imed + neke - 2];
        electron.q2ce_ms0[MXEKE*imed + neke - 1] =
            electron.q2ce_ms0[MXEKE*imed + neke - 2];
        
        /* Now positrons */
        for (int i=0; i<=MXE_SPIN; i++){
            farray[i] = g_array[1*(MXE_SPIN1 + 1) + i];
        }
        for (int i=MXE_SPIN + 1; i<=MXE_SPIN1 - 1; i++){
            farray[i] = g_array[1*(MXE_SPIN1 + 1) + i + 1];
        }
        
        setSpline(elarray, farray, af, bf, cf, df, ndata);
        eil = (1.0 - electron.eke0[imed])/electron.eke1[imed];
        si1e = spline(eil, elarray, af, bf, cf, df, ndata);
        
        for (int i=1; i<=neke - 1; i++){
            eil = (i + 1 - electron.eke0[imed])/electron.eke1[imed];
            si2e = spline(eil, elarray, af, bf, cf, df, ndata);
            electron.q2cp_ms1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron.eke1[imed];
            electron.q2cp_ms0[MXEKE*imed + i - 1] =
                si2e - electron.q2cp_ms1[MXEKE*imed + i - 1]*eil;
            si1e = si2e;
        }
        electron.q2cp_ms1[MXEKE*imed + neke - 1] =
            electron.q2cp_ms1[MXEKE*imed + neke - 2];
        electron.q2cp_ms0[MXEKE*imed + neke - 1] =
            electron.q2cp_ms0[MXEKE*imed + neke - 2];
        
        /* Now substract scattering power that is already taken into account in
         discrete Moller/Bhabha events */
        double tauc = pegs_data.te[imed]/RM;
        int leil;
        double dedx, etap, g_r, g_m, sig;
        si1e=1.0;
        
        for (int i=1; i<=neke - 1; i++){
            eil = (i + 1 - electron.eke0[imed])/electron.eke1[imed];
            e = exp(eil);
            leil = i;
            tau = e/RM;
            
            if (tau > 2.0*tauc){
                sig = electron.esig1[MXEKE*imed + leil]*eil +
                    electron.esig0[MXEKE*imed + leil];
                dedx = electron.ededx1[MXEKE*imed + leil]*eil +
                    electron.ededx0[MXEKE*imed + leil];
                sig /= dedx;
                
                if (sig>1.0E-6) { /* to be sure that this is not a CSDA calc. */
                    etap = electron.etae_ms1[MXEKE*imed + leil]*eil +
                        electron.etae_ms0[MXEKE*imed + leil];
                    eta = 0.25*etap*(electron.xcc[imed])/
                        (electron.blcc[imed])/tau/(tau+2);
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
            
            electron.blcce1[MXEKE*imed + i - 1] =
                (si2e - si1e)*electron.eke1[imed];
            electron.blcce0[MXEKE*imed + i - 1] =
                si2e - electron.blcce1[MXEKE*imed + i - 1]*eil;
            si1e = si2e;
        }
        electron.blcce1[MXEKE*imed + neke - 1] =
            electron.blcce1[MXEKE*imed + neke - 2];
        electron.blcce0[MXEKE*imed + neke - 1] =
            electron.blcce0[MXEKE*imed + neke - 2];
        
    }
    
    fclose(fp);
    
    /* Print information for debugging purposes */
    if(verbose_flag) {
        printf("Listing spin data: \n");
        printf("b2spin_min = %f\n", spin.b2spin_min);
        printf("dbeta2i = %f\n", spin.dbeta2i);
        printf("espml = %f\n", spin.espml);
        printf("dleneri = %f\n", spin.dleneri);
        printf("dqq1i = %f\n", spin.dqq1i);
        printf("\n");
        
        for (int i=0; i<geometry.nmed; i++) {
            printf("For medium %s: \n", geometry.med_names[i]);
            printf("spin_rej = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = i*2*(MXE_SPIN1 + 1)*(MXQ_SPIN + 1)*(MXU_SPIN + 1) + j;
                printf("spin_rej = %f\n", spin.spin_rej[idx]);
            }
            printf("\n");
        }
        
        printf("Listing electron data: \n");
        for (int i=0; i<geometry.nmed; i++) {
            printf("For medium %s: \n", geometry.med_names[i]);
            printf("etae_ms0 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("etae_ms0 = %f\n", electron.etae_ms0[idx]);
            }
            printf("\n");
            
            printf("etae_ms1 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("etae_ms1 = %f\n", electron.etae_ms1[idx]);
            }
            printf("\n");
            
            printf("etap_ms0 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("etap_ms0 = %f\n", electron.etap_ms0[idx]);
            }
            printf("\n");
            
            printf("etap_ms1 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("etap_ms1 = %f\n", electron.etap_ms1[idx]);
            }
            printf("\n");
            
            printf("q1ce_ms0 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("q1ce_ms0 = %f\n", electron.q1ce_ms0[idx]);
            }
            printf("\n");
            
            printf("q1ce_ms1 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("q1ce_ms1 = %f\n", electron.q1ce_ms1[idx]);
            }
            printf("\n");
            
            printf("q1cp_ms0 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("q1cp_ms0 = %f\n", electron.q1cp_ms0[idx]);
            }
            printf("\n");
            
            printf("q1cp_ms1 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("q1cp_ms1 = %f\n", electron.q1cp_ms1[idx]);
            }
            printf("\n");
            
            printf("q2ce_ms0 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("q2ce_ms0 = %f\n", electron.q2ce_ms0[idx]);
            }
            printf("\n");
            
            printf("q2ce_ms1 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("q2ce_ms1 = %f\n", electron.q2ce_ms1[idx]);
            }
            printf("\n");
            
            printf("q2cp_ms0 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("q2cp_ms0 = %f\n", electron.q2cp_ms0[idx]);
            }
            printf("\n");
            
            printf("q2cp_ms1 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("q2cp_ms1 = %f\n", electron.q2cp_ms1[idx]);
            }
            printf("\n");
            
            printf("blcce0 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("blcce0 = %f\n", electron.blcce0[idx]);
            }
            printf("\n");
            
            printf("blcce1 = \n");
            for (int j=0; j<5; j++) { // print just 5 first values
                int idx = MXEKE*i + j;
                printf("blcce1 = %f\n", electron.blcce1[idx]);
            }
            printf("\n");
            
        }
    }
    
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
    
    free(spin.spin_rej);
    
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
