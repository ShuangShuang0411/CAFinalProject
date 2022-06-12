//---------------------------------------------------------------------------------------------------------------------------
// Order of accuracy: 1D acoustic wave problem with the MUSCL-Hancock scheme and PCM, PLM, PPM data reconstruction. (OpenMP)
//---------------------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>


double Gamma = 5.0/3.0;
int nghost = 2;
double T = 2.0;
int DataReconstruct = 2;   // Data reconstruction method: 0 for PCM (constant), 1 for PLM (linear), 2 for PPM (parabolic)
int NThread = 2;   // Total number of threads in OpenMP
int NN [] = {16, 32, 64, 128, 256, 528};


//initial condition: acoustic wave
void InitialCondition_acoustic_wave (int N, double **W ){
    double L     = 1;
    double cs    = 1.0;      // sound speed
    double d_amp = 1.0e-4;    // density perturbation amplitude
    double d0    = 1.0;       // density background
    double u1 = cs*d_amp/d0;        // velocity perturbation
    double P0 = pow(cs, 2.0)*d0/Gamma;   // background pressure
    double P1 = pow(cs, 2.0)*d_amp;      // pressure perturbation
    for (int i=0;i<N;i++){
        W[0][i] = d0 + d_amp*sin(2.0*M_PI*i*L/(N-2*nghost));
        W[1][i] = u1*sin(2.0*M_PI*i*L/(N-2*nghost));
        W[2][i] = P0 + P1*sin(2.0*M_PI*i*L/(N-2*nghost));
    }
}

//boundary condition: periodic
void BoundaryCondition_periodic ( int N, double **U ){
    double **Copy = new double*[3];
    for (int i = 0; i < 3; i++)   Copy[i] = new double[2*nghost];
    for (int i=0; i<nghost; i++){
        Copy[0][i] = U[0][i+N-2*nghost];
        Copy[1][i] = U[1][i+N-2*nghost];
        Copy[2][i] = U[2][i+N-2*nghost];}
    for (int i=nghost; i<2*nghost; i++){
        Copy[0][i] = U[0][i];
        Copy[1][i] = U[1][i];
        Copy[2][i] = U[2][i];}        
    for (int i=0;i<nghost;i++){
        U[0][i] = Copy[0][i];
        U[1][i] = Copy[1][i];
        U[2][i] = Copy[2][i];
        U[0][N-1-i] = Copy[0][2*nghost-i-1];
        U[1][N-1-i] = Copy[1][2*nghost-i-1];
        U[2][N-1-i] = Copy[2][2*nghost-i-1];
    }
}

// primitive variables to conserved variables
void Conserved2Primitive ( int N, double **U, double **pri ){

#   pragma omp parallel for
    for (int i=0;i<N;i++){
        pri[0][i]=U[0][i];
        pri[1][i]=U[1][i]/U[0][i];
        pri[2][i]=(Gamma-1.0)*(U[2][i]-0.5*pow(U[1][i],2)/U[0][i]);
    }
}

// conserved variables to primitive variables
void Primitive2Conserved ( int N, double **pri, double **cons ){

#   pragma omp parallel for    
    for (int i=0;i<N;i++){
        cons[0][i]=pri[0][i];
        cons[1][i]=pri[1][i]*pri[0][i];
        cons[2][i]=pri[2][i]/(Gamma-1)+0.5*pri[0][i]*pow(pri[1][i],2);
    }
}

double ComputePressure( double tho, double px, double e ){
    double p = (Gamma-1.0)*( e - 0.5*pow(px,2)/tho);
    return p;
}

void SoundSpeedMax( int N, double **U, double *s_max) {
    double s[N], p[N];

#   pragma omp parallel for
    for (int i=0;i<N;i++){
        p[i] = ComputePressure(U[0][i],U[1][i],U[2][i]);
        s[i] = sqrt(Gamma*p[i]/U[0][i]);
        if (U[1][i]/U[0][i]>=0)  s[i] += U[1][i]/U[0][i];
        else  s[i] -= U[1][i]/U[0][i];
    }
    *s_max = 0.;
    for (int i=0;i<N;i++){
        if (s[i]>*s_max)  *s_max = s[i];
    }
}

void Conserved2Flux ( int N, double **U, double **flux ){

#   pragma omp parallel for    
    for (int i=0;i<N;i++){
        double P = ComputePressure( U[0][i], U[1][i], U[2][i]);
        double u = U[1][i] / U[0][i];
        flux[0][i] = U[1][i];
        flux[1][i] = u*U[1][i] + P;
        flux[2][i] = (P+U[2][i])*u;
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// PLM and PPM Data Reconstruciton
void ComputeLimitedSlope (int N, double *a, double *slope) {

    double *slope_L = new double [N];
    double *slope_R = new double [N];

//  Apply the van Leer slope limiter
#   pragma omp parallel for
    for (int j=1;j<N-1;j++){
        slope_L[j] = a[j]-a[j-1];
        slope_R[j] = a[j+1]-a[j];
        slope[j] = slope_L[j]*slope_R[j];
        if (slope[j]>0)   slope[j] = 2.0*slope[j]/(slope_L[j]+slope_R[j]);
        else              slope[j] = 0.0;
    }
    slope[0], slope[N-1] = 0.0, 0.0;
}


void ComputeLimitedSlopeCha (int N, double **a, double *cs, double **slope) {

    double **slope_L = new double*[3];
    for (int i = 0; i < 3; i++)   slope_L[i] = new double[N];
    double **slope_R = new double*[3];
    for (int i = 0; i < 3; i++)   slope_R[i] = new double[N];
    double **slope_C = new double*[3];
    for (int i = 0; i < 3; i++)   slope_C[i] = new double[N];
    double **slope_L_cha = new double*[3];
    for (int i = 0; i < 3; i++)   slope_L_cha[i] = new double[N];
    double **slope_R_cha = new double*[3];
    for (int i = 0; i < 3; i++)   slope_R_cha[i] = new double[N];
    double **slope_C_cha = new double*[3];
    for (int i = 0; i < 3; i++)   slope_C_cha[i] = new double[N];
    double **slope_cha = new double*[3];
    for (int i = 0; i < 3; i++)   slope_cha[i] = new double[N];

//  Compute the left-, right-, and centered-differences
    for (int i = 0; i < 3; i++){
        for (int j=1;j<N-1;j++){
            slope_L[i][j] = a[i][j]-a[i][j-1];
            slope_R[i][j] = a[i][j+1]-a[i][j];
            slope_C[i][j] = 0.5*(a[i][j+1]-a[i][j-1]);
        }
    }
//  Project the left, right, and centered differences onto the characteristic variables
    for (int j=1;j<N-1;j++){
        slope_L_cha[0][j] = -0.5*a[0][j]/cs[j]*slope_L[0][j]+0.5*slope_L[2][j]/pow(cs[j],2);
        slope_L_cha[1][j] = slope_L[0][j]-slope_L[2][j]/pow(cs[j],2);
        slope_L_cha[2][j] = 0.5*a[0][j]/cs[j]*slope_L[0][j]+0.5*slope_L[2][j]/pow(cs[j],2);
        slope_R_cha[0][j] = -0.5*a[0][j]/cs[j]*slope_R[0][j]+0.5*slope_R[2][j]/pow(cs[j],2);
        slope_R_cha[1][j] = slope_R[0][j]-slope_R[2][j]/pow(cs[j],2);
        slope_R_cha[2][j] = 0.5*a[0][j]/cs[j]*slope_R[0][j]+0.5*slope_R[2][j]/pow(cs[j],2);
        slope_C_cha[0][j] = -0.5*a[0][j]/cs[j]*slope_C[0][j]+0.5*slope_C[2][j]/pow(cs[j],2);
        slope_C_cha[1][j] = slope_C[0][j]-slope_C[2][j]/pow(cs[j],2);
        slope_C_cha[2][j] = 0.5*a[0][j]/cs[j]*slope_C[0][j]+0.5*slope_C[2][j]/pow(cs[j],2);
    }

//  Apply monotonicity constraints so that the characteristic reconstruction is total variation diminishing (TVD)
    for (int i = 0; i < 3; i++){
        for (int j=1;j<N-1;j++){
            slope_cha[i][j] = std::min(fabs(slope_C_cha[i][j]),std::min(2*fabs(slope_R_cha[i][j]),2*fabs(slope_L_cha[i][j])));
        }
    }
//  Project the monotonized difference in the characteristic variables back onto the primitive variables
    for (int j=1;j<N-1;j++){
        slope[0][j] = slope_cha[0][j] - cs[j]/a[0][j]*slope_cha[1][j] + pow(cs[j],2)*slope_cha[2][j];
        slope[1][j] = slope_cha[0][j];
        slope[2][j] = slope_cha[0][j] + cs[j]/a[0][j]*slope_cha[1][j] + pow(cs[j],2)*slope_cha[2][j];
    }
}


void PLM_Hydro (int N, double **U, double **U_L, double **U_R){

    double **W = new double*[3];
    for (int i = 0; i < 3; i++)   W[i] = new double[N];
    double **W_L = new double*[3];
    for (int i = 0; i < 3; i++)   W_L[i] = new double[N];
    double **W_R = new double*[3];
    for (int i = 0; i < 3; i++)   W_R[i] = new double[N];
    double *slope = new double [N];

    Conserved2Primitive(N, U, W);

    for (int i=0;i<3;i++){
        ComputeLimitedSlope(N, W[i], slope);
#       pragma omp parallel for
        for (int j=1;j<N-1;j++){
//          compute the left and right states of each cell
            W_L[i][j] = W[i][j] - 0.5*slope[j];
            W_R[i][j] = W[i][j] + 0.5*slope[j];
//          ensure face-centered variables lie between nearby volume-averaged (~cell-centered) values
            W_L[i][j] = std::max(W_L[i][j], std::min(W[i][j-1], W[i][j])); 
            W_L[i][j] = std::min(W_L[i][j], std::max(W[i][j-1], W[i][j]));
            W_R[i][j] = 2.0*W[i][j] - W_L[i][j];
            W_R[i][j] = std::max(W_R[i][j], std::min(W[i][j+1], W[i][j])); 
            W_R[i][j] = std::min(W_R[i][j], std::max(W[i][j+1], W[i][j]));
            W_L[i][j] = 2.0*W[i][j] - W_R[i][j];
        }
        W_L[i][0], W_L[i][N-1], W_R[i][0], W_R[i][N-1] = 0.0, 0.0, 0.0, 0.0;
    }

    Primitive2Conserved(N, W_L, U_L);
    Primitive2Conserved(N, W_R, U_R);
}


void interpolation_fnt (int N, int LeftRight, double dx, double *y, double **a, double **a_L, double **a_R, double **value){

    double f = 0.0;
    for (int i=0;i<3;i++){
        for (int j=0;j<N;j++){
            if (LeftRight==0){
                f = a_R[i][j]-y[j]/2/dx*(a_R[i][j]-a_L[i][j]-(1-2/3*y[j]/dx)*6*(a[i][j]-0.5*a_R[i][j]-0.5*a_L[i][j]));
            }
            else{
                f = a_L[i][j]+y[j]/2/dx*(a_R[i][j]-a_L[i][j]+(1-2/3*y[j]/dx)*6*(a[i][j]-0.5*a_R[i][j]-0.5*a_L[i][j]));
            }
            value[i][j] = f;
        }   
    }
}


void PPM_Hydro (int N, double dt, double dx, double **U, double **U_L, double **U_R){

    double **W = new double*[3];
    for (int i = 0; i < 3; i++)   W[i] = new double[N];
    double **W_L = new double*[3];
    for (int i = 0; i < 3; i++)   W_L[i] = new double[N];
    double **W_R = new double*[3];
    for (int i = 0; i < 3; i++)   W_R[i] = new double[N];
    double *slope = new double [N];
//    double **slope = new double*[3];
//    for (int i = 0; i < 3; i++)   slope[i] = new double[N];
    double **W_L_hat = new double*[3];
    for (int i = 0; i < 3; i++)   W_L_hat[i] = new double[N];
    double **W_R_hat = new double*[3];
    for (int i = 0; i < 3; i++)   W_R_hat[i] = new double[N];
    double **W_L_face = new double*[3];
    for (int i = 0; i < 3; i++)   W_L_face[i] = new double[N];
    double **W_R_face = new double*[3];
    for (int i = 0; i < 3; i++)   W_R_face[i] = new double[N];
    double *cs = new double [N];

    Conserved2Primitive(N, U, W);

    for (int j=0;j<N;j++){
        cs[j] = sqrt(Gamma*W[2][j]/W[0][j]);
    }

//    ComputeLimitedSlopeCha(N, W, cs, slope);

    for (int i=0;i<3;i++){
        ComputeLimitedSlope(N, W[i], slope);
#       pragma omp parallel for
        for (int j=1;j<N-1;j++){
//          compute the left and right states of each cell
            W_L[i][j] = 0.5*(W[i][j]+W[i][j-1]) - (slope[j]+slope[j-1])/6.0;
            W_R[i][j] = 0.5*(W[i][j]+W[i][j+1]) - (slope[j]+slope[j+1])/6.0;
        }
        W_L[i][0], W_L[i][N-1], W_R[i][0], W_R[i][N-1] = 0.0, 0.0, 0.0, 0.0;

//      Apply further monotonicity constraints
#       pragma omp parallel for
        for (int j=0;j<N;j++){
            if ( (W_R[i][j]-W[i][j])*(W[i][j]-W_L[i][j])<=0 ) { 
                W_L[i][j] = W[i][j];
                W_R[i][j] = W[i][j];
            }   
            else if ((W_R[i][j]-W_L[i][j])*(W[i][j]-0.5*W_L[i][j]-0.5*W_R[i][j]) > pow(W_R[i][j]-W_L[i][j],2)/6.0){ 
                W_L[i][j] = 3*W[i][j]-2*W_R[i][j];
            }   
            else if (-pow(W_R[i][j]-W_L[i][j],2)/6.0 > (W_R[i][j]-W_L[i][j])*(W[i][j]-0.5*W_L[i][j]-0.5*W_R[i][j])){ 
                W_R[i][j] = 3*W[i][j]-2*W_L[i][j];
            }   
        }
//      ensure face-centered variables lie between nearby volume-averaged (~cell-centered) values
        for (int j=1;j<N-1;j++){
            W_L[i][j] = std::max(W_L[i][j], std::min(W[i][j-1], W[i][j]));
            W_L[i][j] = std::min(W_L[i][j], std::max(W[i][j-1], W[i][j]));
            W_R[i][j] = 2.0*W[i][j] - W_L[i][j];
            W_R[i][j] = std::max(W_R[i][j], std::min(W[i][j+1], W[i][j]));
            W_R[i][j] = std::min(W_R[i][j], std::max(W[i][j+1], W[i][j]));
            W_L[i][j] = 2.0*W[i][j] - W_R[i][j];
        }   
    }

//  Compute the left- and right-interface values using monotonized parabolic interpolation
    double *lambda_p = new double [N];
    double *lambda_n = new double [N];
    double *lambda_0 = new double [N];
    double *lambda_M = new double [N];
    double *lambda_m = new double [N];
    double *lambda_max = new double [N];
    double *lambda_min = new double [N];

    for (int j=0;j<N;j++){
        lambda_p[j] = (W[1][j] + cs[j]);
        lambda_n[j] = (W[1][j] - cs[j]);
        lambda_0[j] = (W[1][j]);
    }

    for (int j=0;j<N;j++){
        lambda_M[j] = std::max(lambda_p[j], std::max(lambda_n[j], lambda_0[j]));
        lambda_m[j] = std::min(lambda_p[j], std::min(lambda_n[j], lambda_0[j]));
    }

    for (int j=0;j<N;j++){
        lambda_max[j] = std::max(0.0, lambda_M[j]);
        lambda_min[j] = std::min(0.0, lambda_m[j]);
        lambda_max[j] *= dt;
        lambda_min[j] *= dt;
    }

    interpolation_fnt(N, 0, dx, lambda_max, W, W_L, W_R, W_L_hat);
    interpolation_fnt(N, 1, dx, lambda_min, W, W_L, W_R, W_R_hat);

//  Perform the characteristic tracing
    double A, B;
    double **W_tep = new double*[3];
    for (int i = 0; i < 3; i++)   W_tep[i] = new double[N];

    for (int j=0;j<N;j++){
        W_L_face[0][j] = W_L_hat[0][j];
        W_L_face[1][j] = W_L_hat[1][j];
        W_L_face[2][j] = W_L_hat[2][j];
    }

    for (int j=0;j<N;j++){
        if (lambda_p[j]>0){
            A = dt/2/dx*(lambda_M[j]-lambda_p[j]);
            B = pow(dt/dx,2)/3*(lambda_M[j]*lambda_M[j]-lambda_p[j]*lambda_p[j]);
            for (int i=0;i<3;i++)   W_tep[i][j] = A*(4*W_R[i][j]+2*W_L[i][j]-6*W[i][j])+B*(6*W[i][j]-3*W_R[i][j]-3*W_L[i][j]);
            W_L_face[0][j] += W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2);
            W_L_face[1][j] += (W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2))*cs[j]/W[0][j];
            W_L_face[2][j] += (W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2))*pow(cs[j],2);
        }
        if (lambda_n[j]>0){
            A = dt/2/dx*(lambda_M[j]-lambda_n[j]);
            B = pow(dt/dx,2)/3*(lambda_M[j]*lambda_M[j]-lambda_n[j]*lambda_n[j]);
            for (int i=0;i<3;i++)   W_tep[i][j] = A*(4*W_R[i][j]+2*W_L[i][j]-6*W[i][j])+B*(6*W[i][j]-3*W_R[i][j]-3*W_L[i][j]);
            W_L_face[0][j] += -W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2);
            W_L_face[1][j] += (-W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2))*-1*cs[j]/W[0][j];
            W_L_face[2][j] += (-W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2))*pow(cs[j],2);
        }
        if (lambda_0[j]>0){
            A = dt/2/dx*(lambda_M[j]-lambda_0[j]);
            B = pow(dt/dx,2)/3*(lambda_M[j]*lambda_M[j]-lambda_0[j]*lambda_0[j]);
            for (int i=0;i<3;i++)   W_tep[i][j] = A*(4*W_R[i][j]+2*W_L[i][j]-6*W[i][j])+B*(6*W[i][j]-3*W_R[i][j]-3*W_L[i][j]);
            W_L_face[0][j] += W_tep[0][j]-W_tep[2][j]/pow(cs[j],2);
        }
    }

    for (int j=0;j<N;j++){
        W_R_face[0][j] = W_R_hat[0][j];
        W_R_face[1][j] = W_R_hat[1][j];
        W_R_face[2][j] = W_R_hat[2][j];
    }

    for (int j=0;j<N;j++){
        if (lambda_p[j]<0){
            A = dt/2/dx*(lambda_m[j]-lambda_p[j]);
            B = pow(dt/dx,2)/3*(lambda_m[j]*lambda_m[j]-lambda_p[j]*lambda_p[j]);
            for (int i=0;i<3;i++)   W_tep[i][j] = A*(6*W[i][j]-2*W_R[i][j]-4*W_L[i][j])+B*(6*W[i][j]-3*W_R[i][j]-3*W_L[i][j]);
            W_R_face[0][j] += W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2);
            W_R_face[1][j] += (W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2))*cs[j]/W[0][j];
            W_R_face[2][j] += (W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2))*pow(cs[j],2);
        }
        if (lambda_n[j]<0){
            A = dt/2/dx*(lambda_m[j]-lambda_n[j]);
            B = pow(dt/dx,2)/3*(lambda_m[j]*lambda_m[j]-lambda_n[j]*lambda_n[j]);
            for (int i=0;i<3;i++)   W_tep[i][j] = A*(6*W[i][j]-2*W_R[i][j]-4*W_L[i][j])+B*(6*W[i][j]-3*W_R[i][j]-3*W_L[i][j]);
            W_R_face[0][j] += -W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2);
            W_R_face[1][j] += (-W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2))*-1*cs[j]/W[0][j];
            W_R_face[2][j] += (-W[0][j]/2/cs[j]*W_tep[1][j]+W_tep[2][j]/2/pow(cs[j],2))*pow(cs[j],2);
        }
        if (lambda_0[j]<0){
            A = dt/2/dx*(lambda_m[j]-lambda_0[j]);
            B = pow(dt/dx,2)/3*(lambda_m[j]*lambda_m[j]-lambda_0[j]*lambda_0[j]);
            for (int i=0;i<3;i++)   W_tep[i][j] = A*(6*W[i][j]-2*W_R[i][j]-4*W_L[i][j])+B*(6*W[i][j]-3*W_R[i][j]-3*W_L[i][j]);
            W_R_face[0][j] += W_tep[0][j]-W_tep[2][j]/pow(cs[j],2);
        }
    }

//  HLL solver correction term
    double **delm_W = new double*[3];
    for (int i = 0; i < 3; i++)   delm_W[i] = new double[N];
    double **corr_L = new double*[3];
    for (int i = 0; i < 3; i++)   corr_L[i] = new double[N];
    double **corr_R = new double*[3];
    for (int i = 0; i < 3; i++)   corr_R[i] = new double[N];

    for (int i=0;i<3;i++){
        for (int j=0;j<N;j++)   delm_W[i][j] = W_R[i][j]-W_L[i][j];
//        for (int j=0;j<N;j++)   delm_W[i][j] = slope[i][j];
    }
    for (int j=0;j<N;j++){
        if (lambda_p[j]<0){
            corr_L[0][j] += (lambda_p[j]-lambda_M[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2));
            corr_L[1][j] += (lambda_p[j]-lambda_M[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*cs[j]/W[0][j];
            corr_L[2][j] += (lambda_p[j]-lambda_M[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*pow(cs[j],2);
        }
        if (lambda_n[j]<0){
            corr_L[0][j] += (lambda_n[j]-lambda_M[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2));
            corr_L[1][j] += (lambda_n[j]-lambda_M[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*(-cs[j])/W[0][j];
            corr_L[2][j] += (lambda_n[j]-lambda_M[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*pow(cs[j],2);
        }
        if (lambda_0[j]<0){
            corr_L[0][j] += (lambda_0[j]-lambda_M[j])*(delm_W[0][j]-delm_W[2][j]/pow(cs[j],2));
        }
        W_L_face[0][j] += -0.5*dt/dx*corr_L[0][j];
        W_L_face[1][j] += -0.5*dt/dx*corr_L[1][j];
        W_L_face[2][j] += -0.5*dt/dx*corr_L[2][j];
    }
    for (int j=0;j<N;j++){
        if (lambda_p[j]>0){
            corr_R[0][j] += (lambda_p[j]-lambda_m[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2));
            corr_R[1][j] += (lambda_p[j]-lambda_m[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*cs[j]/W[0][j];
            corr_R[2][j] += (lambda_p[j]-lambda_m[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*pow(cs[j],2);
        }
        if (lambda_n[j]>0){
            corr_R[0][j] += (lambda_n[j]-lambda_m[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2));
            corr_R[1][j] += (lambda_n[j]-lambda_m[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*(-cs[j])/W[0][j];
            corr_R[2][j] += (lambda_n[j]-lambda_m[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*pow(cs[j],2);
        }
        if (lambda_0[j]>0){
            corr_R[0][j] += (lambda_0[j]-lambda_m[j])*(delm_W[0][j]-delm_W[2][j]/pow(cs[j],2));
        }
        W_R_face[0][j] += -0.5*dt/dx*corr_R[0][j];
        W_R_face[1][j] += -0.5*dt/dx*corr_R[1][j];
        W_R_face[2][j] += -0.5*dt/dx*corr_R[2][j];
    }

    Primitive2Conserved(N, W_R_face, U_L);
    Primitive2Conserved(N, W_L_face, U_R);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////


void HLLC_Riemann_Solver ( int N, double **U_L, double **U_R, double **HLLC_flux ){
    
    double **F_L = new double*[3];
    for (int i = 0; i < 3; i++)   F_L[i] = new double[N]; 
    double **F_R = new double*[3];
    for (int i = 0; i < 3; i++)   F_R[i] = new double[N]; 
    double **F_star_L = new double*[3];
    for (int i = 0; i < 3; i++)   F_star_L[i] = new double[N]; 
    double **F_star_R = new double*[3];
    for (int i = 0; i < 3; i++)   F_star_R[i] = new double[N]; 
    
    double a_L[N], a_R[N];
    double u_L[N], u_R[N];
    double p_R[N], p_L[N], p_star[N];
    double q_L[N], q_R[N], S_L[N], S_R[N], S_star[N];

#   pragma omp parallel for    
    for (int i=0;i<N;i++){
        
        u_L[i] = U_L[1][i]/U_L[0][i];
        u_R[i] = U_R[1][i]/U_R[0][i];
        p_L[i] = ComputePressure(U_L[0][i],U_L[1][i],U_L[2][i]);
        p_R[i] = ComputePressure(U_R[0][i],U_R[1][i],U_R[2][i]);
        a_L[i] = sqrt(Gamma*p_L[i]/U_L[0][i]);
        a_R[i] = sqrt(Gamma*p_R[i]/U_R[0][i]);
        
        //step 1: pressure estimate
        p_star[i] = 0.5*(p_L[i]+p_R[i])-0.5*(u_R[i]-u_L[i])*0.5*(U_L[0][i]+U_R[0][i])*0.5*(a_L[i]+a_R[i]);
        if (p_star[i]<0)   p_star[i] = 0.;
        
        //step 2: wave speed estimate
        if (p_star[i]>p_L[i]){
            q_L[i] = sqrt(1+(Gamma+1)*(p_star[i]/p_L[i]-1)/2.0/Gamma);
        }
        else  q_L[i]=1.0;
        if (p_star[i]>p_R[i]){
            q_R[i] = sqrt(1+(Gamma+1)*(p_star[i]/p_R[i]-1)/2.0/Gamma);
        }
        else  q_R[i]=1.0;
        
        S_L[i] = u_L[i]-a_L[i]*q_L[i];
        S_R[i] = u_R[i]+a_R[i]*q_R[i];
        S_star[i] = (p_R[i]-p_L[i]+U_L[1][i]*(S_L[i]-u_L[i])-U_R[1][i]*(S_R[i]-u_R[i]))/(U_L[0][i]*(S_L[i]-u_L[i])-U_R[0][i]*(S_R[i]-u_R[i]));
        
        //step 3: HLLC flux
        Conserved2Flux(N, U_L, F_L);
        Conserved2Flux(N, U_R, F_R);
        
        F_star_L[0][i] = S_star[i]*(S_L[i]*U_L[0][i]-F_L[0][i])/(S_L[i]-S_star[i]);
        F_star_L[1][i] = (S_star[i]*(S_L[i]*U_L[1][i]-F_L[1][i])+S_L[i]*(p_L[i]+U_L[0][i]*(S_L[i]-u_L[i])*(S_star[i]-u_L[i])))/(S_L[i]-S_star[i]);
        F_star_L[2][i] = (S_star[i]*(S_L[i]*U_L[2][i]-F_L[2][i])+S_L[i]*S_star[i]*(p_L[i]+U_L[0][i]*(S_L[i]-u_L[i])*(S_star[i]-u_L[i])))/(S_L[i]-S_star[i]);
        F_star_R[0][i] = S_star[i]*(S_R[i]*U_R[0][i]-F_R[0][i])/(S_R[i]-S_star[i]);
        F_star_R[1][i] = (S_star[i]*(S_R[i]*U_R[1][i]-F_R[1][i])+S_R[i]*(p_R[i]+U_L[0][i]*(S_R[i]-u_R[i])*(S_star[i]-u_R[i])))/(S_R[i]-S_star[i]);
        F_star_R[2][i] = (S_star[i]*(S_R[i]*U_R[2][i]-F_R[2][i])+S_R[i]*S_star[i]*(p_R[i]+U_L[0][i]*(S_R[i]-u_R[i])*(S_star[i]-u_R[i])))/(S_R[i]-S_star[i]);
        
        if (S_L[i]>=0){
            HLLC_flux[0][i] = F_L[0][i];
            HLLC_flux[1][i] = F_L[1][i];
            HLLC_flux[2][i] = F_L[2][i];
        }
        else if (S_L[i]<=0 && S_star[i]>=0){
            HLLC_flux[0][i] = F_star_L[0][i];
            HLLC_flux[1][i] = F_star_L[1][i];
            HLLC_flux[2][i] = F_star_L[2][i];
        }
        else if (S_star[i]<=0 && S_R[i]>=0){
            HLLC_flux[0][i] = F_star_R[0][i];
            HLLC_flux[1][i] = F_star_R[1][i];
            HLLC_flux[2][i] = F_star_R[2][i];
        }
        else if (S_R[i]<=0){
            HLLC_flux[0][i] = F_R[0][i];
            HLLC_flux[1][i] = F_R[1][i];
            HLLC_flux[2][i] = F_R[2][i];
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main
int main(int argc, const char * argv[]) {

//  OpenMP: Set the number of threads
    omp_set_num_threads( NThread );

    double start;
    double end;

    FILE * data_ptr;
    data_ptr = fopen("./bin/ac_data_evol.txt", "w");
    if (data_ptr==0)  return 0;

    for (int abc=0;abc<6;abc++){  // loop over different N
    start = omp_get_wtime();

    int N = NN[abc];
//    int N = 16;
    double dx = 1.0/(N-2*nghost);

    double **U = new double*[3];
    for (int i = 0; i < 3; i++)   U[i] = new double[N]; 
    double **W = new double*[3];
    for (int i = 0; i < 3; i++)   W[i] = new double[N];
    double **HLLC_flux_L = new double*[3];
    for (int i = 0; i < 3; i++)   HLLC_flux_L[i] = new double[N];
    double **HLLC_flux_R = new double*[3];
    for (int i = 0; i < 3; i++)   HLLC_flux_R[i] = new double[N];
    double **U_L = new double*[3];
    for (int i = 0; i < 3; i++)   U_L[i] = new double[N];
    double **U_R = new double*[3];
    for (int i = 0; i < 3; i++)   U_R[i] = new double[N];
    double **flux_L = new double*[3];
    for (int i = 0; i < 3; i++)   flux_L[i] = new double[N];
    double **flux_R = new double*[3];
    for (int i = 0; i < 3; i++)   flux_R[i] = new double[N];

    double **W_init = new double*[3];
    for (int i = 0; i < 3; i++)   W_init[i] = new double[N];   
    
    int num = 0;
    double dt;
    double t = 0.;
    double S_max = 6.29;
    
    
    InitialCondition_acoustic_wave(N, W);

    for (int i=0;i<3;i++){
        for (int j=0;j<N;j++){
            fprintf(data_ptr,"%e ", W[i][j]);
        }
        fprintf(data_ptr,"\n");
    }

    Primitive2Conserved(N, W, U);
    
    while (t<T){

        BoundaryCondition_periodic(N, U);
//      Compute dt
        SoundSpeedMax(N, U, &S_max);
        dt = dx/S_max;
        t += dt; 
//        printf("Debug: dt = %.10f, t = %.10f\n", dt, t);
        num += 1;

//      MUSCL-Hancock scheme step 1: Data reconstruction
        if (DataReconstruct == 0){   // PCM (constant)
            for (int i=0;i<N;i++){
                U_R[0][i] = U[0][i];
                U_R[1][i] = U[1][i];
                U_R[2][i] = U[2][i];
                U_L[0][i] = U[0][i];
                U_L[1][i] = U[1][i];
                U_L[2][i] = U[2][i];
            }
        }

        else if (DataReconstruct == 1){   // PLM (linear)
            PLM_Hydro (N, U, U_L, U_R);
        }

        else if (DataReconstruct == 2){   // PPM (parabolic)
            PPM_Hydro (N, dt, dx, U, U_L, U_R);
        }

//      MUSCL-Hancock scheme step 2: Evolve the face-centered data by dt/2
        Conserved2Flux(N, U_L, flux_L);
        Conserved2Flux(N, U_R, flux_R);
#       pragma omp parallel for
        for (int i=1;i<N-1;i++){
            U_L[0][i] -= (flux_R[0][i]-flux_L[0][i])*0.5*dt/dx;
            U_L[1][i] -= (flux_R[1][i]-flux_L[1][i])*0.5*dt/dx;
            U_L[2][i] -= (flux_R[2][i]-flux_L[2][i])*0.5*dt/dx;
            U_R[0][i] -= (flux_R[0][i]-flux_L[0][i])*0.5*dt/dx;
            U_R[1][i] -= (flux_R[1][i]-flux_L[1][i])*0.5*dt/dx;
            U_R[2][i] -= (flux_R[2][i]-flux_L[2][i])*0.5*dt/dx;
        }  

        for (int i=N-1;i>0;i--){
            U_R[0][i] = U_R[0][i-1];
            U_R[1][i] = U_R[1][i-1];
            U_R[2][i] = U_R[2][i-1];
        }

//      MUSCL-Hancock scheme step 3: Riemann solver (solve the flux at the left interface)
        HLLC_Riemann_Solver(N, U_R,U_L,HLLC_flux_L);

//      MUSCL-Hancock scheme step 4: Evolve the volume-averaged data by dt
#       pragma omp parallel for
        for (int i=1;i<N-1;i++){
            U[0][i] -= (HLLC_flux_L[0][i+1]-HLLC_flux_L[0][i])*dt/dx;
            U[1][i] -= (HLLC_flux_L[1][i+1]-HLLC_flux_L[1][i])*dt/dx;
            U[2][i] -= (HLLC_flux_L[2][i+1]-HLLC_flux_L[2][i])*dt/dx;
        }

//        BoundaryCondition_periodic(N, U); 
    }
    
    Conserved2Primitive(N, U, W);
    
    end = omp_get_wtime(); 

    InitialCondition_acoustic_wave(N, W_init);
    BoundaryCondition_periodic(N, W_init); 
    double error = 0.0;
    for (int i=nghost;i<N-nghost;i++){
        error += fabs(W_init[0][i] - W[0][i]);
    }
    error /= (N-2*nghost);

    printf("N = %d, DR = %d, total threads = %d\n", N, DataReconstruct, NThread);
    printf("Wall-clock time = %6f, number of iteration = %d\n", end-start, num);    
    printf("Errors = %e\n", error);    

    //save data into file
    for (int i=0;i<3;i++){
        for (int j=0;j<N;j++){
            fprintf(data_ptr,"%e ", W[i][j]);
        }
        fprintf(data_ptr,"\n");
    }

    
    delete[] U;
    delete[] W;
    delete[] HLLC_flux_L;
    delete[] HLLC_flux_R;
    delete[] U_L;
    delete[] U_R;

    } // loop over different N 

    fclose(data_ptr);
    
    return 0;
}

