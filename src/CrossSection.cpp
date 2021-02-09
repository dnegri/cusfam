#include "CrossSection.h"

#define dnst(iso, l)    dnst[l*NISO + iso]

void CrossSection::updateMacroXS(const int& l, float* dnst)
{

	for (int ig = 0; ig < _ng; ig++)
	{
		xsmaca0(ig, l) = 0.0;
		xsmacd0(ig, l) = 0.0;
		xsmacf0(ig, l) = 0.0;
		xsmack0(ig, l) = 0.0;
		xsmacn0(ig, l) = 0.0;
		dpmacd(ig, l) = 0.0;
		dpmaca(ig, l) = 0.0;
		dpmacf(ig, l) = 0.0;
		dpmack(ig, l) = 0.0;
		dpmacn(ig, l) = 0.0;
		ddmacd(ig, l) = 0.0;
		ddmaca(ig, l) = 0.0;
		ddmacf(ig, l) = 0.0;
		ddmack(ig, l) = 0.0;
		ddmacn(ig, l) = 0.0;
		dfmacd(ig, l) = 0.0;
		dfmaca(ig, l) = 0.0;
		dfmacf(ig, l) = 0.0;
		dfmack(ig, l) = 0.0;
		dfmacn(ig, l) = 0.0;

		for (int igs = 0; igs < _ng; igs++)
		{
			xsmacs0(igs, ig, l) = 0.0;;
			dpmacs(igs, ig, l) = 0.0;
			dfmacs(igs, ig, l) = 0.0;
			ddmacs(igs, ig, l) = 0.0;
		}
	}

	for (int ip = 0; ip < NPTM; ip++)
	{
		for (int ig = 0; ig < _ng; ig++)
		{
			dmmacd(ig, ip, l) = 0.0;
			dmmaca(ig, ip, l) = 0.0;
			dmmacf(ig, ip, l) = 0.0;
			dmmack(ig, ip, l) = 0.0;
			dmmacn(ig, ip, l) = 0.0;


			for (int igs = 0; igs < _ng; igs++)
			{
				dmmacs(igs, ig, ip, l) = 0.0;
			}
		}
	}

	for (int i = 0; i < NMAC; i++)
	{
		int iso = ISOMAC[i];

		for (int ig = 0; ig < _ng; ig++)
		{
			xsmaca0(ig, l) = xsmaca0(ig, l) + xsmica0(ig, iso, l) * dnst(iso, l);
			xsmacd0(ig, l) = xsmacd0(ig, l) + xsmicd0(ig, iso, l) * dnst(iso, l);
			xsmacf0(ig, l) = xsmacf0(ig, l) + xsmicf0(ig, iso, l) * dnst(iso, l);
			xsmack0(ig, l) = xsmack0(ig, l) + xsmick0(ig, iso, l) * dnst(iso, l);
			xsmacn0(ig, l) = xsmacn0(ig, l) + xsmicn0(ig, iso, l) * dnst(iso, l);

			dpmacd(ig, l) = dpmacd(ig, l) + xdpmicd(ig, iso, l) * dnst(iso, l);
			dpmaca(ig, l) = dpmaca(ig, l) + xdpmica(ig, iso, l) * dnst(iso, l);
			dpmacf(ig, l) = dpmacf(ig, l) + xdpmicf(ig, iso, l) * dnst(iso, l);
			dpmacn(ig, l) = dpmacn(ig, l) + xdpmicn(ig, iso, l) * dnst(iso, l);
			dpmack(ig, l) = dpmack(ig, l) + xdpmick(ig, iso, l) * dnst(iso, l);
			ddmacd(ig, l) = ddmacd(ig, l) + xddmicd(ig, iso, l) * dnst(iso, l);
			ddmaca(ig, l) = ddmaca(ig, l) + xddmica(ig, iso, l) * dnst(iso, l);
			ddmacf(ig, l) = ddmacf(ig, l) + xddmicf(ig, iso, l) * dnst(iso, l);
			ddmacn(ig, l) = ddmacn(ig, l) + xddmicn(ig, iso, l) * dnst(iso, l);
			ddmack(ig, l) = ddmack(ig, l) + xddmick(ig, iso, l) * dnst(iso, l);
			dfmacd(ig, l) = dfmacd(ig, l) + xdfmicd(ig, iso, l) * dnst(iso, l);
			dfmaca(ig, l) = dfmaca(ig, l) + xdfmica(ig, iso, l) * dnst(iso, l);
			dfmacf(ig, l) = dfmacf(ig, l) + xdfmicf(ig, iso, l) * dnst(iso, l);
			dfmacn(ig, l) = dfmacn(ig, l) + xdfmicn(ig, iso, l) * dnst(iso, l);
			dfmack(ig, l) = dfmack(ig, l) + xdfmick(ig, iso, l) * dnst(iso, l);

			for (int igs = 0; igs < _ng; igs++)
			{
				xsmacs0(igs, ig, l) = xsmacs0(igs, ig, l) + xsmics0(igs, ig, iso, l) * dnst(iso, l);
				dpmacs(igs, ig, l) = dpmacs(igs, ig, l) + xdpmics(igs, ig, iso, l) * dnst(iso, l);
				dfmacs(igs, ig, l) = dfmacs(igs, ig, l) + xdfmics(igs, ig, iso, l) * dnst(iso, l);
				ddmacs(igs, ig, l) = ddmacs(igs, ig, l) + xddmics(igs, ig, iso, l) * dnst(iso, l);

			}
		}

		for (int ip = 0; ip < NPTM; ip++)
		{
			for (int ig = 0; ig < _ng; ig++)
			{
				dmmacd(ig, ip, l) = dmmacd(ig, ip, l) + xdmmicd(ig, ip, iso, l) * dnst(iso, l);
				dmmaca(ig, ip, l) = dmmaca(ig, ip, l) + xdmmica(ig, ip, iso, l) * dnst(iso, l);
				dmmacf(ig, ip, l) = dmmacf(ig, ip, l) + xdmmicf(ig, ip, iso, l) * dnst(iso, l);
				dmmacn(ig, ip, l) = dmmacn(ig, ip, l) + xdmmicn(ig, ip, iso, l) * dnst(iso, l);
				dmmack(ig, ip, l) = dmmack(ig, ip, l) + xdmmick(ig, ip, iso, l) * dnst(iso, l);


				for (int igs = 0; igs < _ng; igs++)
				{
					dmmacs(igs, ig, ip, l) = dmmacs(igs, ig, ip, l) + xdmmics(igs, ig, ip, iso, l) * dnst(iso, l);

				}
			}
		}
	}
}

void CrossSection::updateMacroXS(float* dnst)
{
#pragma omp parallel for
	for (size_t l = 0; l < _nxyz; l++)
	{
		updateMacroXS(l, dnst);
	}
}

void CrossSection::updateXS(const int& l, const float* dnst, const float& dppm, const float& dtf, const float& dtm )
{
	float dtm2 = dtm * dtm;

	for (int ige = 0; ige < _ng; ige++)
	{
		for (int igs = 0; igs < _ng; igs++)
		{
			xssf(igs, ige, l) = xsmacs0(igs, ige, l) + dpmacs(igs, ige, l) * dppm + dfmacs(igs, ige, l) * dtf + dmmacs(igs, ige, 0, l) * dtm + dmmacs(igs, ige, 1, l) * dtm2;
		}
	}

	for (int ig = 0; ig < _ng; ig++)
	{
		xsdf(ig, l) = xsmacd0(ig, l) + dpmacd(ig, l) * dppm + dfmacd(ig, l) * dtf + dmmacd(ig, 0, l) * dtm + dmmacd(ig, 1, l) * dtm2;
		xskf(ig, l) = xsmack0(ig, l) + dpmack(ig, l) * dppm + dfmack(ig, l) * dtf + dmmack(ig, 0, l) * dtm + dmmack(ig, 1, l) * dtm2;
		xsnf(ig, l) = xsmacn0(ig, l) + dpmacn(ig, l) * dppm + dfmacn(ig, l) * dtf + dmmacn(ig, 0, l) * dtm + dmmacn(ig, 1, l) * dtm2;
		xstf(ig, l) = xsmaca0(ig, l) + dpmaca(ig, l) * dppm + dfmaca(ig, l) * dtf + dmmaca(ig, 0, l) * dtm + dmmaca(ig, 1, l) * dtm2;		
	}


	for (int i = 0; i < NNIS; i++)
	{
		int iso = ISONIS[i];

		for (int ig = 0; ig < _ng; ig++)
		{
			xsdf(ig, l) = xsdf(ig, l) + (xsmicd0(ig, iso, l) + xdpmicd(ig, iso, l) * dppm + xdfmicd(ig, iso, l) * dtf + xdmmicd(ig, 0, iso, l) * dtm + xdmmicd(ig, 1, iso, l) * dtm2) * dnst(iso, l);
			xskf(ig, l) = xskf(ig, l) + (xsmick0(ig, iso, l) + xdpmick(ig, iso, l) * dppm + xdfmick(ig, iso, l) * dtf + xdmmick(ig, 0, iso, l) * dtm + xdmmick(ig, 1, iso, l) * dtm2) * dnst(iso, l);
			xsnf(ig, l) = xsnf(ig, l) + (xsmicn0(ig, iso, l) + xdpmicn(ig, iso, l) * dppm + xdfmicn(ig, iso, l) * dtf + xdmmicn(ig, 0, iso, l) * dtm + xdmmicn(ig, 1, iso, l) * dtm2) * dnst(iso, l);
			xstf(ig, l) = xstf(ig, l) + (xsmica0(ig, iso, l) + xdpmica(ig, iso, l) * dppm + xdfmica(ig, iso, l) * dtf + xdmmica(ig, 0, iso, l) * dtm + xdmmica(ig, 1, iso, l) * dtm2) * dnst(iso, l);


			for (int igs = 0; igs < _ng; igs++)
			{
				xssf(igs, ig, l) = xssf(igs, ig, l) + (xsmics0(igs, ig, iso, l) + xdpmics(igs, ig, iso, l) * dppm + xdfmics(igs, ig, iso, l) * dtf + xdmmics(igs, ig, 0, iso, l) * dtm + xdmmics(igs, ig, 1, iso, l) * dtm2) * dnst(iso, l);
			}
		}
	}

	for (int ig = 0; ig < _ng; ig++)
	{
		for (int ige = 0; ige < _ng; ige++)
		{
			if (ig != ige) {
				xstf(ig, l) += xssf(ig, ige, l);
			}
			else {
				xssf(ig, ige, l) = 0.0;
			}
		}
		xsdf(ig, l) = 1. / (3 * xsdf(ig, l));
	}

	// Equilibrium Xenon and depletion
    for (int i = 0; i < NFIS; i++)
    {
        int iso = ISOFIS[i];

        for (int ig = 0; ig < _ng; ig++)
        {
            xsmicf(ig, iso, l) = xsmicf0(ig, iso, l) + xdpmicf(ig, iso, l) * dppm + xdfmicf(ig, iso, l) * dtf + xdmmicf(ig, 0, iso, l) * dtm + xdmmicf(ig, 1, iso, l) * dtm2;
        }
    }

    // Equilibrium Xenon and depletion
    for (int iso = 0; iso < NDEP; iso++)
    {
        for (int ig = 0; ig < _ng; ig++)
        {
            xsmica(ig, iso, l) = xsmica0(ig, iso, l) + xdpmica(ig, iso, l) * dppm + xdfmica(ig, iso, l) * dtf + xdmmica(ig, 0, iso, l) * dtm + xdmmica(ig, 1, iso, l) * dtm2;
        }
    }
}

void CrossSection::updateXS(const float* dnst, const float* dppm, const float* dtf, const float* dtm)
{
    #pragma omp parallel for
	for (size_t l = 0; l < _nxyz; l++)
	{
		updateXS(l, dnst, dppm[l], dtf[l], dtm[l]);
	}

}

