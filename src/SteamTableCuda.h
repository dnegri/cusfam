#pragma once
#include "pch.h"
#include "SteamTable.h"

class SteamTableCuda : public SteamTable {

public:
	__host__ SteamTableCuda(const SteamTable& steam);
	__host__ virtual ~SteamTableCuda();
};


