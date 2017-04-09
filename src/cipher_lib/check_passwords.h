#pragma once

#include <vector>

#include "int_types.h"

typedef std::vector<u8> u8v;
typedef std::vector<u32> u32v;

void CheckPasswords(
    u32* found_password_index_out, u8v& password_blocks, u32 password_length,
    u8v& pbkdf2_salt, u32 pbkdf2_rounds, u8v& blowfish_init_vector,
    u8v& sha1_checksum, u8v& contents_1024);
