#include <cstdio>
#include <cstring>
#include <iostream>

#include "base64.h"
#include "check_passwords.h"
#include "cuda_utilities.h"

using namespace std;

u8v SetupPasswords();

// Config.
const u32 N_PASSWORDS_IN_BATCH(100000);

// Constants.
const u32 N_SHA1_BLOCK_BYTES(64);

int main(int argc, char* argv[])
{
  gpuDeviceInit(0);
  PrintCUDADeviceProperties(0);

  u8v password_blocks(SetupPasswords());

  string pbkdf2_salt_base64("r9GZPGDeM8vIQJDz/pBiQw==");
  string pbkdf2_salt(base64_decode(pbkdf2_salt_base64));
  u8v pbkdf2_salt_v(pbkdf2_salt.begin(), pbkdf2_salt.end());

  string blowfish_init_vector_base64("1M+0j4DeDFw=");
  string blowfish_init_vector(base64_decode(blowfish_init_vector_base64));
  u8v blowfish_init_vector_v(
      blowfish_init_vector.begin(), blowfish_init_vector.end());

  string sha1_checksum_base64("p2PeIegOHaiDJgduyoQr4Q6F8jY=");
  string sha1_checksum(base64_decode(sha1_checksum_base64));
  u8v sha1_checksum_v(sha1_checksum.begin(), sha1_checksum.end());

  string contents_1024_base64(
      "LrioN9+KFLdQh4DmKKqWxGiNWqUxBMhnfqiwYu7VQfOlmMWlmbvUD+/"
      "ZXQ4n5pkMwx7zUstVvs92sqm82Lo2ticCVQojKkj4VFVznePusye0wfUXXZ3Tg4RrLArXmdw"
      "PIDRQcnydZuLi+"
      "Wm1qzmPIKOwtWAqMsS2c2BSQeZpUQaGbmK8QMNyipbcYShi23HS4Mg7oBUZcPXFF+"
      "1sKfDrQRXXpxprwuOesa85zxJG3ufuWTqS/M6B2c9/"
      "D7JeNU4t0qj6HssEO7091YlYOVJWFeMeRM3gMKpVtWJrPosEf48qS1C+"
      "VcTn7UGgJqfE929o49bojsfXqFDCqmSqKkO3FbCyvQM38nUuHLPjzgSntLXMsUx9FSlfxsaq"
      "w2kwcPyyJWnt+qMWtA9IjHprGKo0wWbx0pfQuY46BpDy8BtGEiyc9pvMNmcK+"
      "IaUr9U0nDbF/PjnI8ys+SJfZ6sP++UND3IUtK+XU/EQ/PqqFCkCxU7jb/"
      "EOz3J2cBQ5JWAT99DL/"
      "FUt+"
      "B0lLHKT5uM3CH22A3nBlLfKGw8msRUK2tv8zupgsrlgBvWozR6PoEQYrWyzPytkycR2bfzEl"
      "AJHsK9tXwQ+"
      "1rMqdCdoNjPAT2f6aP4QgjDwkcRrLDLFvhhwBx3JIG92qWajvPAgD1sCxflEpgR3DfHcpkYU"
      "xCDcOjwXP7p2o245Gqe362nosPC3En5XISrveEI9Xl+"
      "IlKszW30y4rRgxoav3Zy49BAx2wYv1EzMK0gVk7ruh3cNB6UZX//"
      "DxDpFPUqsy6uM65V7n3ljmXmJCYEvgacLIKx+1WH5/"
      "e3HbKmjklItPybvEfHM3v6ldI736QQ+UjZwI8x4Fv5lqfL/"
      "e+99QsXZcCIb3NBsDlHx7YH+hCD5qrQ1+5/"
      "h1UasuILlj5sOLF0eyiPSz7lEIGoUWnq0SowjaY5h5IqYqpW591/thSFL7oW/"
      "uIcSVFu76J4eejYvz3vt736wF7L1qhNAM00VTv9aWh7NF3sXUeO9ZYARr0IKixJB8I+"
      "ksTtsiLAMpHFvUM8+2gJcKIjeKvw8SaMY5H82F83vmDvCoq+ypABeYLkcZuWHVpFN61V01U/"
      "4EU9ZpjQhASC+"
      "e77VPL8HQpPkX3xcViuP5t3TGBcay5vysGMMmjqVEtoMrD1h7vT8sXHvQAvlQFUDJo6B518J"
      "+P8/BXvCRg3Va1UrK6Ogi2lkGDREULGMzz8/"
      "zSDlKCQVyci0PXxWApqNFm42qDsChOVe2jHh5tXZyh9V4m6sIcYx4IRcNmiXUT1QfycDeo0R"
      "gioxvXP8Wb0uhvMHkm2URjF/3L2P0vN8RjpzQHP1OXD3/t26h4xeLAk/"
      "iBrCRdrCK2LwI4T6hAOWxnsr6Q==");
  string contents_1024(base64_decode(contents_1024_base64));
  u8v contents_1024_v(contents_1024.begin(), contents_1024.end());

  for (int i(0); i < 100; ++i) {
    u32 found_password;
    CheckPasswords(
        &found_password, password_blocks, 11, pbkdf2_salt_v, 1024,
        blowfish_init_vector_v, sha1_checksum_v, contents_1024_v);
    if (found_password != 0xffffffff) {
      const char* password(
          (const char*)&password_blocks[found_password * N_SHA1_BLOCK_BYTES]);
      cout << "Password found (" << found_password << "): " << password << endl;
      // return 0;
    }
  }

  cudaDeviceReset();
  return 0;
}

// All passwords returned must be of same length.
u8v SetupPasswords()
{
  u8v blocks(N_PASSWORDS_IN_BATCH * N_SHA1_BLOCK_BYTES);

  const char* pw("Xurfers1234");
  u32 pw_len(strlen(pw));
  u8* blocks_ptr(&blocks[0]);
  for (u32 i(0); i < N_PASSWORDS_IN_BATCH; ++i) {
    memcpy(blocks_ptr + i * N_SHA1_BLOCK_BYTES, pw, pw_len);
  }

  blocks[99000 * N_SHA1_BLOCK_BYTES] = (u8)'g';

  return blocks;
}
