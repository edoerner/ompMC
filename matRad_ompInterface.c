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

#define MATLAB_COMPILE_MEX
#include "main.c"

void initPhantomFromMatlab();

//Inputs:
// 1: cube
// 2: stf
// 3: pln
//Outputs:
// Sparse Matrix

#define printf mexPrintf

void mexFunction(
        int nlhs,       mxArray *plhs[], //Output of the function
        int nrhs, const mxArray *prhs[]  //Input of the function
        )
{
    char * tmp;
    
    
    /* Check for proper number of input and output arguments */
    if (nrhs != 4) {
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
    const mxArray * mcOpt = prhs[3];
    
    /* Check data type of input argument 1 / ct cube */
    if (!(mxIsDouble(cubeRho))){
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:inputNotDouble",
                "Input argument must be of type double.");
    }
    
    if (mxGetNumberOfDimensions(cubeRho) != 3){
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:inputNot3D",
                "Input argument 1 must be a three-dimensional cube\n");
    }
    
    /* Check data type of input argument 2 / stf */
    if(!mxIsStruct(mcGeo))
        mexErrMsgIdAndTxt( "MATLAB:phonebook:inputNotStruct",
                "Input 3 must be a mcGeo Structure.");
    
    
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
    
    tmpFieldPointer = mxGetField(mcGeo,0,"nFields");
    
    
    nFields = mxGetScalar(tmpFieldPointer);
    
    tmpFieldPointer = mxGetField(mcGeo,0,"nBixels");
    const double* nBeamletsPerField = mxGetPr(tmpFieldPointer);
    
    unsigned int nBeamlets = 0;
    
    for(int iField=0; iField<nFields; iField++) {
        mexPrintf("%s%d\t%s%d\n","Field ",iField, "Number of Beamlets: ", (unsigned int) nBeamletsPerField[iField]);
        nBeamlets += (unsigned int) nBeamletsPerField[iField];
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
    sprintf(input_items[nInput].key,"global_ecut");
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
    sprintf(input_items[nInput].key,"global_pcut");
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
    sprintf(input_items[nInput].key,"photon xsection");
    tmpFieldPointer = mxGetField(mcOpt,0,"photonXsection");    
    tmp = mxArrayToString(tmpFieldPointer);
    strcpy(input_items[nInput].value,tmp);
    
    input_idx = nInput;
    
    mexPrintf("Input Options:\n");
    for (int iInput = 0; iInput < nInput; iInput++)
        mexPrintf("%s: %s\n",input_items[iInput].key,input_items[iInput].value);
    
    if (verbose_flag)
        mexPrintf("OmpMC output Option: Verbose flag is set!");
    
    
    
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
    
    
    
    
    
    
    
    
    
    double percent_sparse = 0.2;
    //Create Output Matrix
    mwSize nzmax = (mwSize) ceil((double)nCubeElements * percent_sparse);
    
    plhs[0] = mxCreateSparse(nCubeElements,nBeamlets,nzmax,0);
    //sr  = mxGetPr(plhs[0]);
    //si  = mxGetPi(plhs[0]);
    //irs = mxGetIr(plhs[0]);
    //jcs = mxGetJc(plhs[0]);
    
    
    // Run it
    /* Execution time measurement */
    clock_t tbegin, tend;
    tbegin = clock();
}

