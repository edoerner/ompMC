#include <mex.h>
#include <matrix.h>
#include "math.h"
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

#if defined(_WIN32) || defined(_WIN64)
/* We are on Windows */
# define strtok_r strtok_s
#endif

#define exit(EXIT_FAILURE) mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:invalid","Abort.");


#define printf mexPrintf

#define MATLAB_COMPILE_MEX
#include "main.c"

void initPhantomFromMatlab();

//Inputs:
// 1: cube
// 2: stf
// 3: pln
//Outputs:
// Sparse Matrix

void mexFunction(
        int nlhs,       mxArray *plhs[], //Output of the function
        int nrhs, const mxArray *prhs[]  //Input of the function
        )
{
    
    /* Execution time measurement */
    clock_t tbegin, tend;
    tbegin = clock();
    
    char * tmp;
    
    
    /* Check for proper number of input and output arguments */
    if (nrhs != 5) {
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:invalidNumInputs",
                "Two or three input arguments required.");
    }
    if(nlhs > 1){
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:invalidNumOutputs",
                "Too many output arguments.");
    }
    
    
    //Name aliases to allow easier change of number of 
    const mxArray * cubeRho = prhs[0];
    const mxArray * cubeMatIx = prhs[1];
    const mxArray * mcGeo = prhs[2];
    const mxArray * mcSrc = prhs[3];
    const mxArray * mcOpt = prhs[4];
    
    /* Check data type of input argument 1 / ct cube */
    if (!(mxIsDouble(cubeRho))){
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:inputNotDouble",
                "Input argument must be of type double.");
    }
    
    if (mxGetNumberOfDimensions(cubeRho) != 3){
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:inputNot3D",
                "Input argument 1 must be a three-dimensional cube\n");
    }
    
    /* Check data type of input argument 2 / Geo */
    if(!mxIsStruct(mcGeo))
        mexErrMsgIdAndTxt( "MATLAB:phonebook:inputNotStruct",
                "Input 3 must be a mcGeo Structure.");
    
    /* Check data type of input argument 3 / Src */
    if(!mxIsStruct(mcSrc))
        mexErrMsgIdAndTxt( "MATLAB:phonebook:inputNotStruct",
                "Input 4 must be a mcSrc Structure.");
    
    
    //Parse Geometric Information
    /* get input arguments */
    //const char **fnames;       /* pointers to field names */
    //const mwSize *dims;
    //mxArray    *tmp, *fout;
    //char       *pdata=NULL;
    
    unsigned int        nFields;
    int     nGeoStructFields;
    mwSize     ndim,nMaterials;
    mwSize* materialDim;
    mxArray* tmpFieldPointer;//, tmpCellPointer;
    
    
    nGeoStructFields = mxGetNumberOfFields(mcGeo);
    //nFields = (mwSize) mxGetNumberOfElements(mcGeo);
    /* allocate memory  for storing classIDflags */
    //classIDflags = mxCalloc(nGeoStructFields, sizeof(mxClassID));
    
    tmpFieldPointer = mxGetField(mcSrc,0,"nBeams");
    nFields = mxGetScalar(tmpFieldPointer);
    source.nbeams = nFields;
    
    tmpFieldPointer = mxGetField(mcSrc,0,"xSource");
    source.xsource = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"ySource");
    source.ysource = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"zSource");
    source.zsource = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"nBixels");
    const double* nBeamletsPerField = mxGetPr(tmpFieldPointer);
    
    source.nbixels = malloc(source.nbeams*sizeof(int));
    for(int i=0; i<source.nbeams; i++) {
        source.nbixels[i] = (int) nBeamletsPerField[i];
    }
    
    tmpFieldPointer = mxGetField(mcSrc,0,"xCorner");
    source.xcorner = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"yCorner");
    source.ycorner = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"zCorner");
    source.zcorner = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"xSide1");
    source.xside1 = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"ySide1");
    source.yside1 = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"zSide1");
    source.zside1 = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"xSide2");
    source.xside2 = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"ySide2");
    source.yside2 = mxGetPr(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcSrc,0,"zSide2");
    source.zside2 = mxGetPr(tmpFieldPointer);
    
    unsigned int nBeamlets = 0;
    
    for(int iField=0; iField<nFields; iField++) {
        mexPrintf("%s%d\t%s%d\n","Field ",iField, "Number of Beamlets: ", source.nbixels[iField]);
        nBeamlets += (unsigned int) nBeamletsPerField[iField];
        mexPrintf("%s%f %f %f\n", "Source position: ", source.xsource[iField], source.ysource[iField], source.zsource[iField]);
        mexPrintf("%s%f %f %f\n", "Corner position: ", source.xcorner[iField*source.nbixels[iField]], source.ycorner[iField*source.nbixels[iField]], source.zcorner[iField*source.nbixels[iField]]);
        mexPrintf("%s%f %f %f\n", "Side1 position: ", source.xside1[iField*source.nbixels[iField]+1], source.yside1[iField*source.nbixels[iField]+1], source.zside1[iField*source.nbixels[iField]+1]);
        mexPrintf("%s%f %f %f\n", "Side2 position: ", source.xside2[iField*source.nbixels[iField]+1], source.yside2[iField*source.nbixels[iField]+1], source.zside2[iField*source.nbixels[iField]+1]);

    }
    mexPrintf("%s%d\n", "Total Number of Beamlets:", nBeamlets);
    
    //Parse Material
    tmpFieldPointer = mxGetField(mcGeo,0,"material");
    //tmpCellPointer = mxGetPr(tmpFieldPointer);
    materialDim = mxGetDimensions(tmpFieldPointer);
    nMaterials = materialDim[0];
    
    mexPrintf("%s%d\n", "Number of used Materials: ", nMaterials);
    geometry.nmed = nMaterials;
    
    mwIndex tmpSubs[2];
    for (int iMat = 0; iMat < nMaterials; iMat++) {
        tmpSubs[0] = iMat;
        tmpSubs[1] = 0;
        mwSize linIx = mxCalcSingleSubscript(tmpFieldPointer,2,tmpSubs);
        mxArray* tmpCellPointer = mxGetCell(tmpFieldPointer,linIx);
        //mxGetString(tmpCellPointer, geometry.med_names[iMat], mxGetN(tmpCellPointer));
        
        tmp = mxArrayToString(tmpCellPointer);
        if (tmp)
        {
            strcpy(geometry.med_names[iMat],tmp);
            mexPrintf("%s %d:%s\n", "Material", iMat, geometry.med_names[iMat]);
        }
        else
        {
            mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:Error","Material string could not be read!");
        }
    }
    
    //Get the material File!
    //char * materialFileName;
    //tmpFieldPointer = mxGetField(mcGeo,0,"materialFile");
    //mxGetString(tmpFieldPointer,materialFileName,mxGetN(tmpFieldPointer));
    //mexPrintf("%s %s","Material File Used:",materialFileName);
    
     /* Declare variables */
    const mwSize* cubeDim = mxGetDimensions(cubeRho);    
    
    mwSize nCubeElements = cubeDim[0]*cubeDim[1]*cubeDim[2];
    
    geometry.isize = cubeDim[0];
    geometry.jsize = cubeDim[1];
    geometry.ksize = cubeDim[2];
    
    tmpFieldPointer = mxGetField(mcGeo,0,"xBounds");    
    geometry.xbounds = mxGetPr(tmpFieldPointer);
    tmpFieldPointer = mxGetField(mcGeo,0,"yBounds");    
    geometry.ybounds = mxGetPr(tmpFieldPointer);
    tmpFieldPointer = mxGetField(mcGeo,0,"zBounds");    
    geometry.zbounds = mxGetPr(tmpFieldPointer);
    
    geometry.med_densities = mxGetPr(cubeRho);
    if (!mxIsInt32(cubeMatIx))
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:inputNotInt32","The density cube must be a 32 bit integer array!");
    geometry.med_indices = (int*) mxGetPr(cubeMatIx); //Not really safe I would say
    
    
    
    ////////////Parse Options & Create Input Items Structure
    //nOptionsStructFields = mxGetNumberOfFields(mcOpt);
    tmpFieldPointer = mxGetField(mcOpt,0,"verbose");
    bool verbose_flag = mxGetLogicals(tmpFieldPointer)[0];
    
    tmpFieldPointer = mxGetField(mcOpt,0,"nHistories");
    int nhist = (int) mxGetScalar(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcOpt,0,"nBatches");
    int nbatch = (unsigned int) mxGetScalar(tmpFieldPointer);
    
    if (nhist/nbatch == 0) {
        nhist = nbatch;
    }
    
    int nperbatch = nhist/nbatch;
    nhist = nperbatch*nbatch;
    
    mexPrintf("%s: %d\n","Number of Histories", nhist);
    mexPrintf("%s: %d\n","Number of Batches", nbatch);
    
    //Input Items
    //mxArray * tmp = mxCreateString("12345678901234567890123456789012345678901234567890123456789012345678901234567890");
    
    mxArray* tmp2;
    int status;
    int nInput = 0;
    //sprintf(input_items[nInput].key,"spectrum file";
    sprintf(input_items[nInput].key,"spectrum file");
    tmpFieldPointer = mxGetField(mcOpt,0,"spectrumFile");
    tmp = mxArrayToString(tmpFieldPointer);
    strcpy(input_items[nInput].value,tmp);
    
    
    nInput++;
    sprintf(input_items[nInput].key,"mono energy");
    tmpFieldPointer = mxGetField(mcOpt,0,"monoEnergy");
    status = mexCallMATLAB(1, &tmp2, 1,  &tmpFieldPointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }
    
    //mxGetString(tmp,input_items[nInput].value,mxGetN(tmp));
    //input_items[nInput].value = mxArrayToString(tmpFieldPointer);
    //strcpy(input_items[nInput].value,tmp);
    //mexPrintf("%s: %s",input_items[nInput].key,input_items[nInput].value]);
    
    nInput++;
    sprintf(input_items[nInput].key,"collimator bounds");
    tmpFieldPointer = mxGetField(mcOpt,0,"colliBounds");    
    status = mexCallMATLAB(1, &tmp2, 1,  &tmpFieldPointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }

    nInput++;
    sprintf(input_items[nInput].key,"charge");
    tmpFieldPointer = mxGetField(mcOpt,0,"charge");    
    status = mexCallMATLAB(1, &tmp2, 1,  &tmpFieldPointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }

    nInput++;
    sprintf(input_items[nInput].key,"ssd");
    tmpFieldPointer = mxGetField(mcOpt,0,"ssd");    
    status = mexCallMATLAB(1, &tmp2, 1,  &tmpFieldPointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }
    
    nInput++;
    sprintf(input_items[nInput].key,"global ecut");
    tmpFieldPointer = mxGetField(mcOpt,0,"global_ecut");    
    status = mexCallMATLAB(1, &tmp2, 1,  &tmpFieldPointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }

    nInput++;
    sprintf(input_items[nInput].key,"global pcut");
    tmpFieldPointer = mxGetField(mcOpt,0,"global_pcut");    
    status = mexCallMATLAB(1, &tmp2, 1,  &tmpFieldPointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }
    
    nInput++;
    sprintf(input_items[nInput].key,"rng seeds");
    tmpFieldPointer = mxGetField(mcOpt,0,"randomSeeds");    
    status = mexCallMATLAB(1, &tmp2, 1,  &tmpFieldPointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }
    
    nInput++;
    sprintf(input_items[nInput].key,"pegs file");
    tmpFieldPointer = mxGetField(mcOpt,0,"pegsFile");    
    tmp = mxArrayToString(tmpFieldPointer);
    strcpy(input_items[nInput].value,tmp);
    
    nInput++;
    sprintf(input_items[nInput].key,"pgs4form file");
    tmpFieldPointer = mxGetField(mcOpt,0,"pgs4formFile");    
    tmp = mxArrayToString(tmpFieldPointer);
    strcpy(input_items[nInput].value,tmp);
    
    nInput++;
    sprintf(input_items[nInput].key,"data folder");
    tmpFieldPointer = mxGetField(mcOpt,0,"dataFolder");    
    tmp = mxArrayToString(tmpFieldPointer);
    strcpy(input_items[nInput].value,tmp);
    
    nInput++;
    sprintf(input_items[nInput].key,"output folder");
    tmpFieldPointer = mxGetField(mcOpt,0,"outputFolder");    
    tmp = mxArrayToString(tmpFieldPointer);
    strcpy(input_items[nInput].value,tmp);
    
    input_idx = nInput;
    
    mexPrintf("Input Options:\n");
    for (int iInput = 0; iInput < nInput; iInput++)
        mexPrintf("%s: %s\n",input_items[iInput].key,input_items[iInput].value);
    
    if (verbose_flag)
        mexPrintf("OmpMC output Option: Verbose flag is set!");
    
    
    //Read the relative dose threshold
    tmpFieldPointer = mxGetField(mcOpt,0,"relDoseThreshold");
    double relDoseThreshold = mxGetScalar(tmpFieldPointer);
    
    mexPrintf("Using a relative dose cut-off of %f\n",relDoseThreshold);
    
    // Start MC setup
    
    /* Read geometry information from phantom file and initialize geometry */
    //initPhantom();
    
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
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    printf("Total number of particle histories: %d\n", nhist);
    printf("Number of statistical batches: %d\n", nbatch);
    printf("Histories per batch: %d\n", nperbatch);
    
    /* Execution time up to this point */
    printf("Execution time up to this point : %8.5f seconds\n",
           (double)(clock() - tbegin)/CLOCKS_PER_SEC);
    
    double percentage_steps = 0.01; //Steps in which the sparse matrix is allocated
    double percent_sparse = percentage_steps; //Initial percentage to allocate memory for
    //Create Output Matrix
    mwSize nzmax = (mwSize) ceil((double)nCubeElements*(double)nBeamlets * percent_sparse);
    plhs[0] = mxCreateSparse(nCubeElements,nBeamlets,nzmax,mxREAL);
    double *sr  = mxGetPr(plhs[0]);
    mwIndex *irs = mxGetIr(plhs[0]);
    mwIndex *jcs = mxGetJc(plhs[0]);
    mwIndex linIx = 0;
    jcs[0] = 0;
    
    for(int ibeamlet=0; ibeamlet<nBeamlets; ibeamlet++) {
        for (int ibatch=0; ibatch<nbatch; ibatch++) {
//             if (ibatch == 0) {
//                 /* Print header for information during simulation */
//                 printf("%-10s\t%-15s\t%-10s\n", "Batch #", "Elapsed time",
//                        "RNG state");
//                 printf("%-10d\t%-15.5f\t%-5d%-5d\n", ibatch,
//                        (double)(clock() - tbegin)/CLOCKS_PER_SEC, rng.ixx, rng.jxx);
//             }
//             else {
//                 /* Print state of current batch */
//                 printf("%-10d\t%-15.5f\t%-5d%-5d\n", ibatch,
//                        (double)(clock() - tbegin)/CLOCKS_PER_SEC, rng.ixx, rng.jxx);
// 
//             }

            for (int ihist=0; ihist<nperbatch; ihist++) {
                /* Initialize particle history */
                initHistory();

                /* Start electromagnetic shower simulation */
                shower();
            }

            /* Accumulate results of current batch for statistical analysis */
            accumEndep();
        }
        
        int iout = 1;   /* i.e. deposit mean dose per particle fluence */
        //outputResults("output_dose", iout, nhist, nbatch);
        accumulateResults(iout, nhist, nbatch);

        //Get maximum value to apply threshold
        double doseMax = 0.0;
        for (int irl=1; irl < gridsize+1; irl++)
        {
            if (score.accum_endep[irl] > doseMax)
                doseMax = score.accum_endep[irl];
        }
        double thresh = doseMax * relDoseThreshold;

//         mexPrintf("Found maximum dose value of %.3e Gy, applying threshold of %.3e Gy.\n",doseMax,thresh);

        //Count values above threshold
        mwSize nnz = 0; //Number of nonzeros in the dose cube

        for (int irl=1; irl < gridsize+1; irl++)
        {        
            if (score.accum_endep[irl] > thresh)
                nnz++;
        }
//         mexPrintf("Found %d significant values, equals %f percent of whole cube.\n",nnz,100.0* (double) nnz / (double) gridsize);

        //Check if we need to reallocate for sparse matrix
        if ((linIx + nnz) > nzmax)
        {
            int oldnzmax = nzmax;
            percent_sparse += percentage_steps;
            
            nzmax = (mwSize) ceil((double)nCubeElements*(double)nBeamlets*percent_sparse);
            
            
            /* Make sure nzmax increases atleast by 1. */
            if (oldnzmax == nzmax)
                nzmax++;
            
            if (verbose_flag)
                mexPrintf("Reallocating Sparse Matrix from nzmax=%d to nzmax=%d\n",oldnzmax,nzmax);
            
            //Set new nzmax and reallocate more memory
            mxSetNzmax(plhs[0], nzmax);
            mxSetPr(plhs[0], mxRealloc(sr, nzmax*sizeof(double)));
            mxSetIr(plhs[0], mxRealloc(irs, nzmax*sizeof(int)));
            
            //Use the new pointers
            sr  = mxGetPr(plhs[0]);
            irs = mxGetIr(plhs[0]);
        }
        
        //double *sr  = mxCalloc(nnz,sizeof(double));
        //mwIndex *irs = mxCalloc(nnz,sizeof(mwIndex));
        //mwIndex *jcs = mxCalloc(nnz,sizeof(

        for (int irl=1; irl < gridsize+1; irl++)
        {        
            if (score.accum_endep[irl] > thresh) {            
                sr[linIx] = score.accum_endep[irl];
                irs[linIx] = irl-1;
                //mexPrintf("Element %d: Index %d and value %.3e",linIx,irs[linIx],sr[linIx]);
                linIx++;
            }
        }
        
        jcs[ibeamlet+1] = linIx;
//         for (mwIndex iBeamlet = 2; iBeamlet <= nBeamlets; iBeamlet++)
//             jcs[iBeamlet] = linIx;
        
        /* Reset accum_endep for following beamlet */
        memset(score.accum_endep, 0.0, (gridsize + 1)*sizeof(double));
        
    }
    
    mexPrintf("Sparse MC Dij has %d (%f percent) elements!\n",linIx,(double) linIx / ((double)nCubeElements*(double)nBeamlets));
    
    //Truncate the matrix to the exact size by reallocation
    mxSetNzmax(plhs[0], linIx);
    mxSetPr(plhs[0], mxRealloc(sr, linIx*sizeof(double)));
    mxSetIr(plhs[0], mxRealloc(irs, linIx*sizeof(int)));
    
    sr  = mxGetPr(plhs[0]);
    irs = mxGetIr(plhs[0]);
    
    /* Print some output and execution time up to this point */
    mexPrintf("Simulation finished\n");
    mexPrintf("Execution time up to this point : %8.5f seconds\n",
           (double)(clock() - tbegin)/CLOCKS_PER_SEC);
    
    /* Analysis and output of results */
//     if (verbose_flag) {
//         /* Sum energy deposition in the phantom */
//         double etot = 0.0;
//         for (int irl=1; irl<gridsize+1; irl++) {
//             etot += score.accum_endep[irl];
//         }
//         printf("Fraction of incident energy deposited in the phantom: %5.4f\n",
//                etot/score.ensrc);
//     }
    
    //mxSetNzmax(plhs[0],nnz);
    //mxSetPr(plhs[0],sr);
    //mxSetIr(plhs[0],irs);
    //mxSetJc(plhs[0],jcs);
    
    /* Cleaning */
    /*cleanPhantom();*/
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
        
    
    
}

