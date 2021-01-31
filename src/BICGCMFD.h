#pragma once
#include "CMFD.h"
#include "BICGSolver.h"

class BICGCMFD : public CMFD {
private:
    int _nmaxbicg;
    double _epsbicg;
    double* _eshift_diag;
    BICGSolver* _ls;
private:
    double& eshift_diag(const int& igs, const int& ige, const int& l) {return _eshift_diag[l*_g.ng2()+ige*_g.ng()+igs];};
public:
    BICGCMFD(Geometry &g, CrossSection &x);
    virtual ~BICGCMFD();

    void upddtil() override;
    void upddhat(double* flux, double* jnet) override;
    void setls() override;
    void updls(const double& reigvs);
    void updjnet(double* flux, double* jnet) override;
    void setls(const int &l);

    void updls(const int& l, const double& reigvs);

    void drive(double& eigv, double* flux, double* psi, float& errl2);
    double residual(const double& reigv, const double& reigvs, double* flux, double* psi);
    void axb(double* flux, double* aflux);
    double wiel(const int& icy, double* flux, double* psi, double& eigv, double& reigv, double& reigvs);



};


