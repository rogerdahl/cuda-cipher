// Stdlib.
#include "stdio.h"
#include <iostream>
#include <stdint.h>
#include <string.h>
#include <vector>

// CUDA
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// App.
#include "check_passwords.h"
#include "cuda_utilities.h"
#include "int_types.h"

using namespace std;
using namespace thrust;

// Host.

__global__ void CheckPasswordsKernel(
    u32* found_password_index_out, u8* password_blocks, u32 n_passwords,
    u32 password_len, u8* pbkdf2_salt, u32 pbkdf2_salt_len, u32 pbkdf2_rounds,
    u32* blowfish_init_vector, u32* sha1_checksum, u8* contents_section,
    u32 contents_section_len);

int select_threads_per_block();

// PBKDF2-HMAC-SHA1

__device__ void PBKDF2(
    u8* key_out, const u8* password, u32 password_len, const u8* salt,
    u32 salt_len, u32 key_len, u32 rounds);

// HMAC-SHA1

__device__ void HMACSHA1(
    u32* hash_out, const void* key, u32 key_len, const void* msg, u32 msg_len);
__device__ void HMACSHA1InnerOuter(
    u32* hash_inner_out, u32* hash_outer_out, const void* key, u32 key_len);
__device__ void HMACSHA1Hash(
    u32* hash_out, u32* hash_inner, u32* hash_outer, const void* msg);
__device__ void HMACSHA1HashToBlock(u32* hash);

const u32 HMAC_INNER_PADDING(0x36363636);
const u32 HMAC_OUTER_PADDING(0x5c5c5c5c);

// SHA1

__device__ void SHA1InitState(u32* state);
__device__ void SHA1HashMessage(u32* hash, u8* msg, u32 len);
__device__ void SHA1CompressFullBlock(u32* state, const u32* block);
__device__ void SHA1CompressFullBlockHMACInner(u32* state, const u32* block);
__device__ void SHA1CompressFullBlockHMACOuter(u32* state, const u32* block);
__device__ void SHA1CompressLastBlock(u32* state, u8* msg, u32 len);
__device__ void SHA1CompressLastBlockEmpty(u32* state, u32 len);
__device__ void SHA1StateToHash(u32* hash, u32* state);
__device__ void SHA1CopyHash(u32* dst, u32* src);
__device__ bool SHA1CompareHash(u32* hash1, u32* hash2);
__device__ void SHA1SwapEndian(u32* hash);

const u32 N_SHA1_HASH_BYTES(20);
const u32 N_SHA1_HASH_WORDS(N_SHA1_HASH_BYTES / sizeof(u32));
const u32 N_SHA1_BLOCK_BYTES(64);
const u32 N_SHA1_BLOCK_WORDS(N_SHA1_BLOCK_BYTES / sizeof(u32));

// Blowfish

const u32 N_BLOWFISH_BLOCK_BYTES(8);
const u32 N_BLOWFISH_ROUNDS(16);

struct BlowfishSubkey
{
  u32 s0[256];
  u32 s1[256];
  u32 s2[256];
  u32 s3[256];
  u32 p[N_BLOWFISH_ROUNDS + 2];
};

__device__ void BlowfishCreateSubkey(
    BlowfishSubkey& subkey_out, u8* key, u32 key_len);
__device__ void BlowfishEncrypt(
    BlowfishSubkey& subkey, u32* left_out, u32* right_out);
//__device__ void BlowfishDecrypt(BlowfishSubkey& subkey, u32* left_out, u32*
//right_out); // not in use
__device__ void BlowfishDecryptBufferCFB(
    BlowfishSubkey& subkey, u32* buf_out, u32* buf_in, u32 n_blocks, u32* iv);

// Util, device

__device__ u32 SwapEndian32(u32 x);
__device__ void SwapEndian32Ptr(u32* x);
__device__ u32 Min(u32 a, u32 b);

// Util, host

void HostSwapEndian32Ptr(u32* x);
u32v u8vToLittleEndianu32v(u8v&);

// Given an array of passwords, check if any of them match the valid hash.
// All passwords in the array must have the same size.
void CheckPasswords(
    u32* found_password_index_out, u8v& password_blocks, u32 password_len,
    u8v& pbkdf2_salt, u32 pbkdf2_rounds, u8v& blowfish_init_vector,
    u8v& sha1_checksum, u8v& contents_section)
{
  // Copy passwords to device.
  host_vector<u8> password_blocks_h(
      password_blocks.begin(), password_blocks.end());
  device_vector<u8> password_blocks_d = password_blocks_h;
  u8* password_blocks_ptr_d(raw_pointer_cast(&password_blocks_d[0]));
  u32 n_passwords(password_blocks.size() / N_SHA1_BLOCK_BYTES);

  // Copy PBKDF2 salt to device (zero padded to N_SHA1_BLOCK_BYTES).
  host_vector<u8> pbkdf2_salt_h(pbkdf2_salt.begin(), pbkdf2_salt.end());
  pbkdf2_salt_h.resize(N_SHA1_BLOCK_BYTES);
  device_vector<u8> pbkdf2_salt_d = pbkdf2_salt_h;
  u8* pbkdf2_salt_ptr_d(raw_pointer_cast(&pbkdf2_salt_d[0]));

  // Convert Blowfish initialization vector to 2 32-bit little endian words and
  // copy to device.
  host_vector<u32> blowfish_init_vector_h(
      u8vToLittleEndianu32v(blowfish_init_vector));
  device_vector<u32> blowfish_init_vector_d = blowfish_init_vector_h;
  u32* blowfish_init_vector_ptr_d(raw_pointer_cast(&blowfish_init_vector_d[0]));

  // Convert SHA1 checksum to 5 32-bit little endian words and copy to device.
  host_vector<u32> sha1_checksum_h(u8vToLittleEndianu32v(sha1_checksum));
  // host_vector<u32> sha1_checksum_h(sha1_checksum);
  device_vector<u32> sha1_checksum_d = sha1_checksum_h;
  u32* sha1_checksum_ptr_d(raw_pointer_cast(&sha1_checksum_d[0]));

  // Copy first 1024 bytes of encrypted contents.xml to device.
  host_vector<u8> contents_section_h(
      contents_section.begin(), contents_section.end());
  device_vector<u8> contents_section_d = contents_section_h;
  u8* contents_section_ptr_d(raw_pointer_cast(&contents_section_d[0]));

  // Buffer for index of found password. Single 32 bit word.
  // device_vector<u32> found_password_index_out_d(1, 0xffffffff);
  // u32*
  // found_password_index_out_ptr_d(raw_pointer_cast(&found_password_index_out_d[0]));
  // device_vector bug workaround (crash when setting to fixed size or doing
  // resize)
  u32* found_password_index_out_ptr_d;
  u32 not_found(0xffffffff);
  cudaCheck(cudaMalloc(&found_password_index_out_ptr_d, 4));
  cudaCheck(
      cudaMemcpy(
          found_password_index_out_ptr_d, &not_found, sizeof(not_found),
          cudaMemcpyHostToDevice));

  // Run kernel.

  // threads_per_block should be obtained from the occupancy calculator.

  //  for (u32 threads_per_block_x = 32; threads_per_block_x <= 1024;
  //  threads_per_block_x += 32) {
  u32 threads_per_block_x = select_threads_per_block();
  u32 threads_per_block_y(1);

  // Dimension of each thread block (number of threads to launch in each block).
  dim3 block_dim(threads_per_block_x, threads_per_block_y);

  // Dimension of the grid (number of blocks to launch).
  dim3 grid_dim(DivUp(n_passwords, threads_per_block_x));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaFuncSetCacheConfig(CheckPasswordsKernel, cudaFuncCachePreferL1);

  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);

  CheckPasswordsKernel<<<grid_dim, block_dim>>>(
      found_password_index_out_ptr_d, password_blocks_ptr_d, n_passwords,
      password_len, pbkdf2_salt_ptr_d, pbkdf2_salt.size(), pbkdf2_rounds,
      blowfish_init_vector_ptr_d, sha1_checksum_ptr_d, contents_section_ptr_d,
      contents_section.size());

  cudaCheckLastError("CheckPasswordsKernel");

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("time (ms): %f\n", elapsedTime);
  //}
  // host_vector<u32> found_password_index_out_h = found_password_index_out_d;
  //*found_password_index_out = found_password_index_out_h[0];
  // device_vector bug workaround (crash when setting to fixed size or doing
  // resize)
  cudaCheck(
      cudaMemcpy(
          found_password_index_out, found_password_index_out_ptr_d, 4,
          cudaMemcpyDeviceToHost));
}

int select_threads_per_block()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  u32 threads_per_block = 0;

  // GeForce GTX 570
  if (prop.major == 2 && prop.minor == 0 && prop.multiProcessorCount == 15) {
    threads_per_block = 256;
  }
  // GeForce GTX 660 (3.0)
  else if (prop.major == 3 && prop.minor == 0 && prop.multiProcessorCount == 5) {
    threads_per_block = 256;
  }
  // GeForce GTX 780 Ti (3.5)
  else if (prop.major == 3 && prop.minor == 5 && prop.multiProcessorCount == 15) {
    threads_per_block = 768;
  }
  // GeForce GTX 750 Ti (5.0)
  else if (prop.major == 5 && prop.minor == 0 && prop.multiProcessorCount == 5) {
    threads_per_block = 512;
  }
  else {
    threads_per_block = 512;
  }
  printf("threads per block: %d\n", threads_per_block);

  return threads_per_block;
}

// passwords: array of passwords. Each is zero padded out to 64 bytes.
// key: 64 bytes, zero padded
// target_sha1_hash: 5 u32 words.
// found_password_idx: 1 u32 word (output)
__global__ void CheckPasswordsKernel(
    u32* found_password_index_out, u8* password_blocks, u32 n_passwords,
    u32 password_len, u8* pbkdf2_salt, u32 pbkdf2_salt_len, u32 pbkdf2_rounds,
    u32* blowfish_init_vector, u32* sha1_checksum, u8* contents_section,
    u32 contents_section_len)
{
  u32 i(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_passwords) {
    return;
  }

  // Get SHA1 hash of the password.
  u32 hash[N_SHA1_HASH_WORDS];
  SHA1HashMessage(hash, password_blocks + i * N_SHA1_BLOCK_BYTES, password_len);
  SHA1SwapEndian(hash);

  // Use PBKDF2 key extender to derive Blowfish key from password and salt.
  u8 key[16];
  PBKDF2(
      key, (u8*)hash, N_SHA1_HASH_BYTES, pbkdf2_salt, pbkdf2_salt_len,
      sizeof(key), pbkdf2_rounds);

  //// For benchmarking only the PBKDF2-HMAC-SHA1, make sure the compiler can't
  ///drop the PBKDF2 calculation.
  //// Only stores k in the extremely unlikely case that the 128-bit key is all
  ///zeroes.
  // u8 k = 0;
  // for (int j = 0; j < 16; ++j) {
  //  k |= key[j];
  //}
  // if (!k) {
  //  *found_password_index_out = i;
  //}
  // return;

  // Blowfish decrypt first 1024 bytes of contents.xml with blowfish key and
  // initialization vector.
  BlowfishSubkey subkey;
  BlowfishCreateSubkey(subkey, key, sizeof(key));
  u32 n_blowfish_blocks(contents_section_len / N_BLOWFISH_BLOCK_BYTES);
  u32 contents_section_decrypted[1024 / sizeof(u32)];
  BlowfishDecryptBufferCFB(
      subkey, contents_section_decrypted, (u32*)contents_section,
      n_blowfish_blocks, (u32*)blowfish_init_vector);

  // Get SHA1 hash of decrypted version of first 1024 bytes of contents.xml.
  SHA1HashMessage(hash, (u8*)contents_section_decrypted, contents_section_len);

  // If SHA1 hash matches the one supplied in the OpenOffice document manifest,
  // the password
  // has been found.
  if (SHA1CompareHash(hash, sha1_checksum)) {
    *found_password_index_out = i;
  }
}

///////////////////////////////////////////////////////////////////////////
// PBKDF2

// Inner loop: YES
__device__ void PBKDF2(
    u8* key_out, const u8* password, u32 password_len, const u8* salt,
    u32 salt_len, u32 key_len, u32 rounds)
{
  u32 inner[N_SHA1_HASH_WORDS];
  u32 outer[N_SHA1_HASH_WORDS];
  HMACSHA1InnerOuter(inner, outer, password, password_len);

  u8 salt_local
      [32]; /////////////////////////////////////////////////////////////////////////////
            ///arbitrary
  memcpy(salt_local, salt, salt_len);

  for (int count(1); key_len > 0; ++count) {
    salt_local[salt_len + 0] = (count >> 24) & 0xff;
    salt_local[salt_len + 1] = (count >> 16) & 0xff;
    salt_local[salt_len + 2] = (count >> 8) & 0xff;
    salt_local[salt_len + 3] = count & 0xff;

    u32 state[64 / sizeof(u32)];
    HMACSHA1(state, password, password_len, salt_local, salt_len + 4);
    u32 obuf[N_SHA1_HASH_WORDS];
    SHA1CopyHash(obuf, state);
    HMACSHA1HashToBlock(state);

    // INNER LOOP.
    for (int i(1); i < rounds; ++i) {
      HMACSHA1Hash(state, inner, outer, state);
      for (int j(0); j < N_SHA1_HASH_WORDS; ++j) {
        obuf[j] ^= state[j];
      }
    }

    u32 r(Min(key_len, N_SHA1_HASH_BYTES));
    memcpy(key_out, obuf, r);
    key_out += r;
    key_len -= r;
  };
}

///////////////////////////////////////////////////////////////////////////
// HMAC-SHA1

// Creates a SHA1 hash from a message and a key. The difference from
// regular SHA1 is the presense of the key.
//
// hash_out: The generated SHA1 hash. 20 bytes.
// key must be a 5 * 4 (20) byte hash.
// key_len must be <= N_SHA1_BLOCK_BYTES.
// key = password
// msg = salt (first round), sha1_hash (remaining rounds)
// len of openoffice password = 1 or more bytes
// len of openoffice salt = 16 bytes
// len of sha1_hash = 20 bytes
//
// Inner loop: NO
__device__ void HMACSHA1(
    u32* hash_out, const void* key, u32 key_len, const void* msg, u32 msg_len)
{
  u32 state[N_SHA1_HASH_WORDS];

  u8 __align__(4) buffer[N_SHA1_BLOCK_BYTES];
  memset(buffer, 0, N_SHA1_BLOCK_BYTES);
  memcpy(buffer, key, key_len);

  SHA1InitState(state);
  SHA1CompressFullBlockHMACInner(state, (u32*)buffer);

  while (msg_len >= N_SHA1_BLOCK_BYTES) {
    SHA1CompressFullBlock(state, (u32*)msg);
    msg = (u8*)msg + N_SHA1_BLOCK_BYTES;
    msg_len -= N_SHA1_BLOCK_BYTES;
  }

  // N_SHA1_BLOCK_BYTES is added to msg_len because message is being
  // (virtually) appended to i_key_pad.
  SHA1CompressLastBlock(state, (u8*)msg, N_SHA1_BLOCK_BYTES + msg_len);

  u32 hash[N_SHA1_HASH_WORDS];
  SHA1StateToHash(hash, state);
  SHA1InitState(hash_out);
  SHA1CompressFullBlockHMACOuter(hash_out, (u32*)buffer);
  SHA1CompressLastBlock(hash_out, (u8*)hash, N_SHA1_BLOCK_BYTES + 5 * 4);
  SHA1SwapEndian(hash_out);
}

// Optimization: Allows computation of the some hashes to be moved outside of
// the inner loop in PBKDF2.
// Inner loop: NO
__device__ void HMACSHA1InnerOuter(
    u32* hash_inner_out, u32* hash_outer_out, const void* key, u32 key_len)
{
  u8 __align__(4) buffer[N_SHA1_BLOCK_BYTES];
  memset(buffer, 0, N_SHA1_BLOCK_BYTES);
  memcpy(buffer, key, key_len);

  SHA1InitState(hash_inner_out);
  SHA1CompressFullBlockHMACInner(hash_inner_out, (u32*)buffer);

  SHA1InitState(hash_outer_out);
  SHA1CompressFullBlockHMACOuter(hash_outer_out, (u32*)buffer);
}

// Optimized for msg being an SHA1 hash (20 bytes).
// Inner loop: YES
__device__ void HMACSHA1Hash(
    u32* hash_out, u32* hash_inner, u32* hash_outer, const void* msg)
{
  u32 state[N_SHA1_HASH_WORDS];
  SHA1CopyHash(state, hash_inner);
  SHA1CompressFullBlock(state, reinterpret_cast<const u32*>(msg));
  u32 hash[64 / sizeof(u32)];
  SHA1StateToHash(hash, state);
  SHA1CopyHash(hash_out, hash_outer);
  HMACSHA1HashToBlock(hash);
  SHA1CompressFullBlock(hash_out, reinterpret_cast<u32*>(hash));
  SHA1SwapEndian(hash_out);
}

// Prepare a 64 byte block that contains a 20 byte SHA1 hash to be hashed as
// the last block in a two-block SHA1 message.
// Inner loop: YES
__device__ void HMACSHA1HashToBlock(u32* hash)
{
  // Set message terminating bit.
  hash[5] = 0x00000080; // SwapEndian32(0x80000000);
  hash[6] = 0x00000000;
  hash[7] = 0x00000000;
  hash[8] = 0x00000000;
  hash[9] = 0x00000000;
  hash[10] = 0x00000000;
  hash[11] = 0x00000000;
  hash[12] = 0x00000000;
  hash[13] = 0x00000000;
  // Set message size (1st block 64 bytes, second block 20 bytes = 84 bytes =
  // 672 bits = 02a0)
  hash[14] = 0x00000000;
  hash[15] = 0xa0020000; // SwapEndian32(0x000002a0);
}

///////////////////////////////////////////////////////////////////////////////
// SHA1

// Inner loop: YES
__device__ void SHA1InitState(u32* state)
{
  state[0] = 0x67452301;
  state[1] = 0xefcdab89;
  state[2] = 0x98badcfe;
  state[3] = 0x10325476;
  state[4] = 0xc3d2e1f0;
}

// Inner loop: NO
__device__ void SHA1HashMessage(u32* hash, u8* msg, u32 len)
{
  SHA1InitState(hash);

  for (int i(0); i + N_SHA1_BLOCK_BYTES <= len; i += N_SHA1_BLOCK_BYTES) {
    SHA1CompressFullBlock(hash, reinterpret_cast<u32*>(msg + i));
  }

  SHA1CompressLastBlock(hash, msg, len);

  // SHA1SwapEndian(hash);
}

// An SHA1 block is 16 32-bit words = 64 bytes = 512 bits.
// An SHA1 hash is 5 32-bit words = 20 bytes = 160 bits.
#define SCHEDULE(i)                                            \
  tmp = schedule[(i - 3) & 0xf] ^ schedule[(i - 8) & 0xf]      \
        ^ schedule[(i - 14) & 0xf] ^ schedule[(i - 16) & 0xf]; \
  schedule[i & 0xf] = tmp << 1 | tmp >> 31;
#define R0A(a, b, c, d, e, i)                                    \
  schedule[i] = (block[i] << 24) | ((block[i] & 0xff00) << 8)    \
                | ((block[i] >> 8) & 0xff00) | (block[i] >> 24); \
  RTAIL(a, b, e, ((b & c) | (~b & d)), i, 0x5a827999)
#define R0B(a, b, c, d, e, i) \
  SCHEDULE(i) RTAIL(a, b, e, ((b & c) | (~b & d)), i, 0x5a827999)
#define R1(a, b, c, d, e, i) \
  SCHEDULE(i) RTAIL(a, b, e, (b ^ c ^ d), i, 0x6ed9eba1)
#define R2(a, b, c, d, e, i) \
  SCHEDULE(i) RTAIL(a, b, e, ((b & c) ^ (b & d) ^ (c & d)), i, 0x8f1bbcdc)
#define R3(a, b, c, d, e, i) \
  SCHEDULE(i) RTAIL(a, b, e, (b ^ c ^ d), i, 0xca62c1d6)
#define RTAIL(a, b, e, f, i, k)                        \
  e += (a << 5 | a >> 27) + f + k + schedule[i & 0xf]; \
  b = b << 30 | b >> 2;
#define RS()                                                                                \
  R0B(e, a, b, c, d, 16)                                                                    \
  R0B(d, e, a, b, c, 17) R0B(c, d, e, a, b, 18) R0B(b, c, d, e, a, 19) R1(                  \
      a, b, c, d, e, 20) R1(e, a, b, c, d, 21) R1(d, e, a, b, c, 22)                        \
      R1(c, d, e, a, b, 23) R1(b, c, d, e, a, 24) R1(a, b, c, d, e, 25) R1(                 \
          e, a, b, c, d, 26) R1(d, e, a, b, c, 27) R1(c, d, e, a, b, 28)                    \
          R1(b, c, d, e, a, 29) R1(a, b, c, d, e, 30) R1(                                   \
              e,                                                                            \
              a,                                                                            \
              b,                                                                            \
              c,                                                                            \
              d,                                                                            \
              31) R1(d, e, a, b, c, 32)                                                     \
              R1(c, d, e, a, b, 33) R1(                                                     \
                  b,                                                                        \
                  c,                                                                        \
                  d,                                                                        \
                  e,                                                                        \
                  a,                                                                        \
                  34) R1(a, b, c, d, e, 35)                                                 \
                  R1(e, a, b, c, d, 36) R1(                                                 \
                      d,                                                                    \
                      e,                                                                    \
                      a,                                                                    \
                      b,                                                                    \
                      c,                                                                    \
                      37) R1(c, d, e, a, b, 38)                                             \
                      R1(b, c, d, e, a, 39) R2(                                             \
                          a,                                                                \
                          b,                                                                \
                          c,                                                                \
                          d,                                                                \
                          e,                                                                \
                          40) R2(e, a, b, c, d, 41)                                         \
                          R2(d, e, a, b, c, 42) R2(                                         \
                              c,                                                            \
                              d,                                                            \
                              e,                                                            \
                              a,                                                            \
                              b,                                                            \
                              43) R2(b, c, d, e, a, 44)                                     \
                              R2(a, b, c, d, e, 45) R2(                                     \
                                  e,                                                        \
                                  a,                                                        \
                                  b,                                                        \
                                  c,                                                        \
                                  d,                                                        \
                                  46) R2(d, e, a, b, c, 47)                                 \
                                  R2(c, d, e, a, b, 48) R2(b, c, d, e, a, 49) R2(           \
                                      a,                                                    \
                                      b,                                                    \
                                      c,                                                    \
                                      d,                                                    \
                                      e,                                                    \
                                      50) R2(e, a, b, c, d, 51)                             \
                                      R2(d, e, a, b, c, 52) R2(c, d, e, a, b, 53) R2(       \
                                          b,                                                \
                                          c,                                                \
                                          d,                                                \
                                          e,                                                \
                                          a,                                                \
                                          54) R2(a, b, c, d, e, 55)                         \
                                          R2(e, a, b, c, d, 56) R2(d, e, a, b, c, 57) R2(   \
                                              c,                                            \
                                              d,                                            \
                                              e,                                            \
                                              a,                                            \
                                              b,                                            \
                                              58) R2(b, c, d, e, a, 59)                     \
                                              R3(a, b, c, d, e, 60) R3(                     \
                                                  e,                                        \
                                                  a,                                        \
                                                  b,                                        \
                                                  c,                                        \
                                                  d,                                        \
                                                  61) R3(d, e, a, b, c, 62)                 \
                                                  R3(c, d, e, a, b, 63) R3(                 \
                                                      b,                                    \
                                                      c,                                    \
                                                      d,                                    \
                                                      e,                                    \
                                                      a,                                    \
                                                      64) R3(a, b, c, d, e, 65)             \
                                                      R3(e, a, b, c, d, 66) R3(             \
                                                          d,                                \
                                                          e,                                \
                                                          a,                                \
                                                          b,                                \
                                                          c,                                \
                                                          67) R3(c, d, e, a, b, 68)         \
                                                          R3(b, c, d, e, a, 69) R3(         \
                                                              a,                            \
                                                              b,                            \
                                                              c,                            \
                                                              d,                            \
                                                              e,                            \
                                                              70) R3(e, a, b, c, d, 71)     \
                                                              R3(d, e, a, b, c, 72) R3(     \
                                                                  c,                        \
                                                                  d,                        \
                                                                  e,                        \
                                                                  a,                        \
                                                                  b,                        \
                                                                  73) R3(b, c, d, e, a, 74) \
                                                                  R3(a, b, c, d, e, 75) R3( \
                                                                      e,                    \
                                                                      a,                    \
                                                                      b,                    \
                                                                      c,                    \
                                                                      d,                    \
                                                                      76)                   \
                                                                      R3(d, e,              \
                                                                         a, b,              \
                                                                         c,                 \
                                                                         77)                \
                                                                          R3(c,             \
                                                                             d,             \
                                                                             e,             \
                                                                             a,             \
                                                                             b,             \
                                                                             78)            \
                                                                              R3(b,         \
                                                                                 c,         \
                                                                                 d,         \
                                                                                 e,         \
                                                                                 a,         \
                                                                                 79)

// Inner loop: YES
__device__ void SHA1CompressFullBlock(u32* state, const u32* block)
{
  u32 a(state[0]);
  u32 b(state[1]);
  u32 c(state[2]);
  u32 d(state[3]);
  u32 e(state[4]);
  u32 schedule[16];
  u32 tmp;
  R0A(a, b, c, d, e, 0)
  R0A(e, a, b, c, d, 1) R0A(d, e, a, b, c, 2) R0A(c, d, e, a, b, 3)
      R0A(b, c, d, e, a, 4) R0A(a, b, c, d, e, 5) R0A(e, a, b, c, d, 6)
          R0A(d, e, a, b, c, 7) R0A(c, d, e, a, b, 8) R0A(b, c, d, e, a, 9)
              R0A(a, b, c, d, e, 10) R0A(e, a, b, c, d, 11)
                  R0A(d, e, a, b, c, 12) R0A(c, d, e, a, b, 13)
                      R0A(b, c, d, e, a, 14) R0A(a, b, c, d, e, 15) RS()
                          state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
}

#define R0AHMACI(a, b, c, d, e, i)                                          \
  tmp = block[i] ^ HMAC_INNER_PADDING;                                      \
  schedule[i] = (tmp << 24) | ((tmp & 0xff00) << 8) | ((tmp >> 8) & 0xff00) \
                | (tmp >> 24);                                              \
  RTAIL(a, b, e, ((b & c) | (~b & d)), i, 0x5a827999)

// Inner loop: NO
__device__ void SHA1CompressFullBlockHMACInner(u32* state, const u32* block)
{
  u32 a(state[0]);
  u32 b(state[1]);
  u32 c(state[2]);
  u32 d(state[3]);
  u32 e(state[4]);
  u32 schedule[16];
  u32 tmp;
  R0AHMACI(a, b, c, d, e, 0)
  R0AHMACI(e, a, b, c, d, 1) R0AHMACI(d, e, a, b, c, 2)
      R0AHMACI(c, d, e, a, b, 3) R0AHMACI(b, c, d, e, a, 4)
          R0AHMACI(a, b, c, d, e, 5) R0AHMACI(e, a, b, c, d, 6)
              R0AHMACI(d, e, a, b, c, 7) R0AHMACI(c, d, e, a, b, 8)
                  R0AHMACI(b, c, d, e, a, 9) R0AHMACI(a, b, c, d, e, 10)
                      R0AHMACI(e, a, b, c, d, 11) R0AHMACI(d, e, a, b, c, 12)
                          R0AHMACI(c, d, e, a, b, 13)
                              R0AHMACI(b, c, d, e, a, 14)
                                  R0AHMACI(a, b, c, d, e, 15) RS() state[0] +=
      a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
}

#define R0AHMACO(a, b, c, d, e, i)                                          \
  tmp = block[i] ^ HMAC_OUTER_PADDING;                                      \
  schedule[i] = (tmp << 24) | ((tmp & 0xff00) << 8) | ((tmp >> 8) & 0xff00) \
                | (tmp >> 24);                                              \
  RTAIL(a, b, e, ((b & c) | (~b & d)), i, 0x5a827999)

// Inner loop: NO
__device__ void SHA1CompressFullBlockHMACOuter(u32* state, const u32* block)
{
  u32 a(state[0]);
  u32 b(state[1]);
  u32 c(state[2]);
  u32 d(state[3]);
  u32 e(state[4]);
  u32 schedule[16];
  u32 tmp;
  R0AHMACO(a, b, c, d, e, 0)
  R0AHMACO(e, a, b, c, d, 1) R0AHMACO(d, e, a, b, c, 2)
      R0AHMACO(c, d, e, a, b, 3) R0AHMACO(b, c, d, e, a, 4)
          R0AHMACO(a, b, c, d, e, 5) R0AHMACO(e, a, b, c, d, 6)
              R0AHMACO(d, e, a, b, c, 7) R0AHMACO(c, d, e, a, b, 8)
                  R0AHMACO(b, c, d, e, a, 9) R0AHMACO(a, b, c, d, e, 10)
                      R0AHMACO(e, a, b, c, d, 11) R0AHMACO(d, e, a, b, c, 12)
                          R0AHMACO(c, d, e, a, b, 13)
                              R0AHMACO(b, c, d, e, a, 14)
                                  R0AHMACO(a, b, c, d, e, 15) RS() state[0] +=
      a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
}

// Process the last, partial block and message length.
// The SHA1 spec requires the message to end with a "1" bit and then 8 bytes
// with the length of the message, in bits.
// Inner loop: NO
__device__ void SHA1CompressLastBlock(u32* state, u8* msg, u32 len)
{
  u32 block[N_SHA1_BLOCK_WORDS];
  u8* byteBlock = (u8*)block;

  int rem = len % 64;
  memcpy(byteBlock, msg, rem);

  byteBlock[rem] = 0x80;
  rem++;
  if (N_SHA1_BLOCK_BYTES - rem >= 8) {
    memset(byteBlock + rem, 0, 56 - rem);
  }
  else {
    memset(byteBlock + rem, 0, N_SHA1_BLOCK_BYTES - rem);
    SHA1CompressFullBlock(state, block);
    memset(block, 0, 56);
  }

  u64 longLen = ((u64)len) << 3;
  for (int i = 0; i < 8; i++) {
    byteBlock[N_SHA1_BLOCK_BYTES - 1 - i] = (u8)(longLen >> (i * 8));
  }
  SHA1CompressFullBlock(state, block);
}

// Special case for hashing the last block, when the last block is empty.
// This becomes the last step in hashing any message of length that is a
// multiple of 64 bytes.
// Inner loop: NO
__device__ void SHA1CompressLastBlockEmpty(u32* state, u32 len)
{
  u32 block[N_SHA1_BLOCK_WORDS] = { 0 };
  u8* byteBlock = (u8*)block;
  byteBlock[0] = 0x80;

  u64 long_len(len * 8);
  for (int i(0); i < 8; i++) {
    byteBlock[N_SHA1_BLOCK_BYTES - 1 - i] = (u8)(long_len >> (i * 8));
  }
  SHA1CompressFullBlock(state, block);
}

// Inner loop: YES
__device__ void SHA1StateToHash(u32* hash, u32* state)
{
  for (int i(0); i < N_SHA1_HASH_WORDS; ++i) {
    hash[i] = SwapEndian32(state[i]);
  }
}

// Inner loop: YES
__device__ void SHA1CopyHash(u32* dst, u32* src)
{
  for (int i(0); i < N_SHA1_HASH_WORDS; ++i) {
    dst[i] = src[i];
  }
}

__device__ bool SHA1CompareHash(u32* hash1, u32* hash2)
{
  return !(
      hash1[0] != hash2[0] || hash1[1] != hash2[1] || hash1[2] != hash2[2]
      || hash1[3] != hash2[3] || hash1[4] != hash2[4]);
}

// Inner loop: YES
__device__ void SHA1SwapEndian(u32* hash)
{
  for (int i(0); i < N_SHA1_HASH_WORDS; ++i) {
    SwapEndian32Ptr(hash + i);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Blowfish

// Substitution-boxes.
__constant__ const u32 ks0[256] =
    { 0xD1310BA6, 0x98DFB5AC, 0x2FFD72DB, 0xD01ADFB7, 0xB8E1AFED, 0x6A267E96,
      0xBA7C9045, 0xF12C7F99, 0x24A19947, 0xB3916CF7, 0x0801F2E2, 0x858EFC16,
      0x636920D8, 0x71574E69, 0xA458FEA3, 0xF4933D7E, 0x0D95748F, 0x728EB658,
      0x718BCD58, 0x82154AEE, 0x7B54A41D, 0xC25A59B5, 0x9C30D539, 0x2AF26013,
      0xC5D1B023, 0x286085F0, 0xCA417918, 0xB8DB38EF, 0x8E79DCB0, 0x603A180E,
      0x6C9E0E8B, 0xB01E8A3E, 0xD71577C1, 0xBD314B27, 0x78AF2FDA, 0x55605C60,
      0xE65525F3, 0xAA55AB94, 0x57489862, 0x63E81440, 0x55CA396A, 0x2AAB10B6,
      0xB4CC5C34, 0x1141E8CE, 0xA15486AF, 0x7C72E993, 0xB3EE1411, 0x636FBC2A,
      0x2BA9C55D, 0x741831F6, 0xCE5C3E16, 0x9B87931E, 0xAFD6BA33, 0x6C24CF5C,
      0x7A325381, 0x28958677, 0x3B8F4898, 0x6B4BB9AF, 0xC4BFE81B, 0x66282193,
      0x61D809CC, 0xFB21A991, 0x487CAC60, 0x5DEC8032, 0xEF845D5D, 0xE98575B1,
      0xDC262302, 0xEB651B88, 0x23893E81, 0xD396ACC5, 0x0F6D6FF3, 0x83F44239,
      0x2E0B4482, 0xA4842004, 0x69C8F04A, 0x9E1F9B5E, 0x21C66842, 0xF6E96C9A,
      0x670C9C61, 0xABD388F0, 0x6A51A0D2, 0xD8542F68, 0x960FA728, 0xAB5133A3,
      0x6EEF0B6C, 0x137A3BE4, 0xBA3BF050, 0x7EFB2A98, 0xA1F1651D, 0x39AF0176,
      0x66CA593E, 0x82430E88, 0x8CEE8619, 0x456F9FB4, 0x7D84A5C3, 0x3B8B5EBE,
      0xE06F75D8, 0x85C12073, 0x401A449F, 0x56C16AA6, 0x4ED3AA62, 0x363F7706,
      0x1BFEDF72, 0x429B023D, 0x37D0D724, 0xD00A1248, 0xDB0FEAD3, 0x49F1C09B,
      0x075372C9, 0x80991B7B, 0x25D479D8, 0xF6E8DEF7, 0xE3FE501A, 0xB6794C3B,
      0x976CE0BD, 0x04C006BA, 0xC1A94FB6, 0x409F60C4, 0x5E5C9EC2, 0x196A2463,
      0x68FB6FAF, 0x3E6C53B5, 0x1339B2EB, 0x3B52EC6F, 0x6DFC511F, 0x9B30952C,
      0xCC814544, 0xAF5EBD09, 0xBEE3D004, 0xDE334AFD, 0x660F2807, 0x192E4BB3,
      0xC0CBA857, 0x45C8740F, 0xD20B5F39, 0xB9D3FBDB, 0x5579C0BD, 0x1A60320A,
      0xD6A100C6, 0x402C7279, 0x679F25FE, 0xFB1FA3CC, 0x8EA5E9F8, 0xDB3222F8,
      0x3C7516DF, 0xFD616B15, 0x2F501EC8, 0xAD0552AB, 0x323DB5FA, 0xFD238760,
      0x53317B48, 0x3E00DF82, 0x9E5C57BB, 0xCA6F8CA0, 0x1A87562E, 0xDF1769DB,
      0xD542A8F6, 0x287EFFC3, 0xAC6732C6, 0x8C4F5573, 0x695B27B0, 0xBBCA58C8,
      0xE1FFA35D, 0xB8F011A0, 0x10FA3D98, 0xFD2183B8, 0x4AFCB56C, 0x2DD1D35B,
      0x9A53E479, 0xB6F84565, 0xD28E49BC, 0x4BFB9790, 0xE1DDF2DA, 0xA4CB7E33,
      0x62FB1341, 0xCEE4C6E8, 0xEF20CADA, 0x36774C01, 0xD07E9EFE, 0x2BF11FB4,
      0x95DBDA4D, 0xAE909198, 0xEAAD8E71, 0x6B93D5A0, 0xD08ED1D0, 0xAFC725E0,
      0x8E3C5B2F, 0x8E7594B7, 0x8FF6E2FB, 0xF2122B64, 0x8888B812, 0x900DF01C,
      0x4FAD5EA0, 0x688FC31C, 0xD1CFF191, 0xB3A8C1AD, 0x2F2F2218, 0xBE0E1777,
      0xEA752DFE, 0x8B021FA1, 0xE5A0CC0F, 0xB56F74E8, 0x18ACF3D6, 0xCE89E299,
      0xB4A84FE0, 0xFD13E0B7, 0x7CC43B81, 0xD2ADA8D9, 0x165FA266, 0x80957705,
      0x93CC7314, 0x211A1477, 0xE6AD2065, 0x77B5FA86, 0xC75442F5, 0xFB9D35CF,
      0xEBCDAF0C, 0x7B3E89A0, 0xD6411BD3, 0xAE1E7E49, 0x00250E2D, 0x2071B35E,
      0x226800BB, 0x57B8E0AF, 0x2464369B, 0xF009B91E, 0x5563911D, 0x59DFA6AA,
      0x78C14389, 0xD95A537F, 0x207D5BA2, 0x02E5B9C5, 0x83260376, 0x6295CFA9,
      0x11C81968, 0x4E734A41, 0xB3472DCA, 0x7B14A94A, 0x1B510052, 0x9A532915,
      0xD60F573F, 0xBC9BC6E4, 0x2B60A476, 0x81E67400, 0x08BA6FB5, 0x571BE91F,
      0xF296EC6B, 0x2A0DD915, 0xB6636521, 0xE7B9F9B6, 0xFF34052E, 0xC5855664,
      0x53B02D5D, 0xA99F8FA1, 0x08BA4799, 0x6E85076A };

__constant__ const u32 ks1[256] =
    { 0x4B7A70E9, 0xB5B32944, 0xDB75092E, 0xC4192623, 0xAD6EA6B0, 0x49A7DF7D,
      0x9CEE60B8, 0x8FEDB266, 0xECAA8C71, 0x699A17FF, 0x5664526C, 0xC2B19EE1,
      0x193602A5, 0x75094C29, 0xA0591340, 0xE4183A3E, 0x3F54989A, 0x5B429D65,
      0x6B8FE4D6, 0x99F73FD6, 0xA1D29C07, 0xEFE830F5, 0x4D2D38E6, 0xF0255DC1,
      0x4CDD2086, 0x8470EB26, 0x6382E9C6, 0x021ECC5E, 0x09686B3F, 0x3EBAEFC9,
      0x3C971814, 0x6B6A70A1, 0x687F3584, 0x52A0E286, 0xB79C5305, 0xAA500737,
      0x3E07841C, 0x7FDEAE5C, 0x8E7D44EC, 0x5716F2B8, 0xB03ADA37, 0xF0500C0D,
      0xF01C1F04, 0x0200B3FF, 0xAE0CF51A, 0x3CB574B2, 0x25837A58, 0xDC0921BD,
      0xD19113F9, 0x7CA92FF6, 0x94324773, 0x22F54701, 0x3AE5E581, 0x37C2DADC,
      0xC8B57634, 0x9AF3DDA7, 0xA9446146, 0x0FD0030E, 0xECC8C73E, 0xA4751E41,
      0xE238CD99, 0x3BEA0E2F, 0x3280BBA1, 0x183EB331, 0x4E548B38, 0x4F6DB908,
      0x6F420D03, 0xF60A04BF, 0x2CB81290, 0x24977C79, 0x5679B072, 0xBCAF89AF,
      0xDE9A771F, 0xD9930810, 0xB38BAE12, 0xDCCF3F2E, 0x5512721F, 0x2E6B7124,
      0x501ADDE6, 0x9F84CD87, 0x7A584718, 0x7408DA17, 0xBC9F9ABC, 0xE94B7D8C,
      0xEC7AEC3A, 0xDB851DFA, 0x63094366, 0xC464C3D2, 0xEF1C1847, 0x3215D908,
      0xDD433B37, 0x24C2BA16, 0x12A14D43, 0x2A65C451, 0x50940002, 0x133AE4DD,
      0x71DFF89E, 0x10314E55, 0x81AC77D6, 0x5F11199B, 0x043556F1, 0xD7A3C76B,
      0x3C11183B, 0x5924A509, 0xF28FE6ED, 0x97F1FBFA, 0x9EBABF2C, 0x1E153C6E,
      0x86E34570, 0xEAE96FB1, 0x860E5E0A, 0x5A3E2AB3, 0x771FE71C, 0x4E3D06FA,
      0x2965DCB9, 0x99E71D0F, 0x803E89D6, 0x5266C825, 0x2E4CC978, 0x9C10B36A,
      0xC6150EBA, 0x94E2EA78, 0xA5FC3C53, 0x1E0A2DF4, 0xF2F74EA7, 0x361D2B3D,
      0x1939260F, 0x19C27960, 0x5223A708, 0xF71312B6, 0xEBADFE6E, 0xEAC31F66,
      0xE3BC4595, 0xA67BC883, 0xB17F37D1, 0x018CFF28, 0xC332DDEF, 0xBE6C5AA5,
      0x65582185, 0x68AB9802, 0xEECEA50F, 0xDB2F953B, 0x2AEF7DAD, 0x5B6E2F84,
      0x1521B628, 0x29076170, 0xECDD4775, 0x619F1510, 0x13CCA830, 0xEB61BD96,
      0x0334FE1E, 0xAA0363CF, 0xB5735C90, 0x4C70A239, 0xD59E9E0B, 0xCBAADE14,
      0xEECC86BC, 0x60622CA7, 0x9CAB5CAB, 0xB2F3846E, 0x648B1EAF, 0x19BDF0CA,
      0xA02369B9, 0x655ABB50, 0x40685A32, 0x3C2AB4B3, 0x319EE9D5, 0xC021B8F7,
      0x9B540B19, 0x875FA099, 0x95F7997E, 0x623D7DA8, 0xF837889A, 0x97E32D77,
      0x11ED935F, 0x16681281, 0x0E358829, 0xC7E61FD6, 0x96DEDFA1, 0x7858BA99,
      0x57F584A5, 0x1B227263, 0x9B83C3FF, 0x1AC24696, 0xCDB30AEB, 0x532E3054,
      0x8FD948E4, 0x6DBC3128, 0x58EBF2EF, 0x34C6FFEA, 0xFE28ED61, 0xEE7C3C73,
      0x5D4A14D9, 0xE864B7E3, 0x42105D14, 0x203E13E0, 0x45EEE2B6, 0xA3AAABEA,
      0xDB6C4F15, 0xFACB4FD0, 0xC742F442, 0xEF6ABBB5, 0x654F3B1D, 0x41CD2105,
      0xD81E799E, 0x86854DC7, 0xE44B476A, 0x3D816250, 0xCF62A1F2, 0x5B8D2646,
      0xFC8883A0, 0xC1C7B6A3, 0x7F1524C3, 0x69CB7492, 0x47848A0B, 0x5692B285,
      0x095BBF00, 0xAD19489D, 0x1462B174, 0x23820E00, 0x58428D2A, 0x0C55F5EA,
      0x1DADF43E, 0x233F7061, 0x3372F092, 0x8D937E41, 0xD65FECF1, 0x6C223BDB,
      0x7CDE3759, 0xCBEE7460, 0x4085F2A7, 0xCE77326E, 0xA6078084, 0x19F8509E,
      0xE8EFD855, 0x61D99735, 0xA969A7AA, 0xC50C06C2, 0x5A04ABFC, 0x800BCADC,
      0x9E447A2E, 0xC3453484, 0xFDD56705, 0x0E1E9EC9, 0xDB73DBD3, 0x105588CD,
      0x675FDA79, 0xE3674340, 0xC5C43465, 0x713E38D8, 0x3D28F89E, 0xF16DFF20,
      0x153E21E7, 0x8FB03D4A, 0xE6E39F2B, 0xDB83ADF7 };

__constant__ const u32 ks2[256] =
    { 0xE93D5A68, 0x948140F7, 0xF64C261C, 0x94692934, 0x411520F7, 0x7602D4F7,
      0xBCF46B2E, 0xD4A20068, 0xD4082471, 0x3320F46A, 0x43B7D4B7, 0x500061AF,
      0x1E39F62E, 0x97244546, 0x14214F74, 0xBF8B8840, 0x4D95FC1D, 0x96B591AF,
      0x70F4DDD3, 0x66A02F45, 0xBFBC09EC, 0x03BD9785, 0x7FAC6DD0, 0x31CB8504,
      0x96EB27B3, 0x55FD3941, 0xDA2547E6, 0xABCA0A9A, 0x28507825, 0x530429F4,
      0x0A2C86DA, 0xE9B66DFB, 0x68DC1462, 0xD7486900, 0x680EC0A4, 0x27A18DEE,
      0x4F3FFEA2, 0xE887AD8C, 0xB58CE006, 0x7AF4D6B6, 0xAACE1E7C, 0xD3375FEC,
      0xCE78A399, 0x406B2A42, 0x20FE9E35, 0xD9F385B9, 0xEE39D7AB, 0x3B124E8B,
      0x1DC9FAF7, 0x4B6D1856, 0x26A36631, 0xEAE397B2, 0x3A6EFA74, 0xDD5B4332,
      0x6841E7F7, 0xCA7820FB, 0xFB0AF54E, 0xD8FEB397, 0x454056AC, 0xBA489527,
      0x55533A3A, 0x20838D87, 0xFE6BA9B7, 0xD096954B, 0x55A867BC, 0xA1159A58,
      0xCCA92963, 0x99E1DB33, 0xA62A4A56, 0x3F3125F9, 0x5EF47E1C, 0x9029317C,
      0xFDF8E802, 0x04272F70, 0x80BB155C, 0x05282CE3, 0x95C11548, 0xE4C66D22,
      0x48C1133F, 0xC70F86DC, 0x07F9C9EE, 0x41041F0F, 0x404779A4, 0x5D886E17,
      0x325F51EB, 0xD59BC0D1, 0xF2BCC18F, 0x41113564, 0x257B7834, 0x602A9C60,
      0xDFF8E8A3, 0x1F636C1B, 0x0E12B4C2, 0x02E1329E, 0xAF664FD1, 0xCAD18115,
      0x6B2395E0, 0x333E92E1, 0x3B240B62, 0xEEBEB922, 0x85B2A20E, 0xE6BA0D99,
      0xDE720C8C, 0x2DA2F728, 0xD0127845, 0x95B794FD, 0x647D0862, 0xE7CCF5F0,
      0x5449A36F, 0x877D48FA, 0xC39DFD27, 0xF33E8D1E, 0x0A476341, 0x992EFF74,
      0x3A6F6EAB, 0xF4F8FD37, 0xA812DC60, 0xA1EBDDF8, 0x991BE14C, 0xDB6E6B0D,
      0xC67B5510, 0x6D672C37, 0x2765D43B, 0xDCD0E804, 0xF1290DC7, 0xCC00FFA3,
      0xB5390F92, 0x690FED0B, 0x667B9FFB, 0xCEDB7D9C, 0xA091CF0B, 0xD9155EA3,
      0xBB132F88, 0x515BAD24, 0x7B9479BF, 0x763BD6EB, 0x37392EB3, 0xCC115979,
      0x8026E297, 0xF42E312D, 0x6842ADA7, 0xC66A2B3B, 0x12754CCC, 0x782EF11C,
      0x6A124237, 0xB79251E7, 0x06A1BBE6, 0x4BFB6350, 0x1A6B1018, 0x11CAEDFA,
      0x3D25BDD8, 0xE2E1C3C9, 0x44421659, 0x0A121386, 0xD90CEC6E, 0xD5ABEA2A,
      0x64AF674E, 0xDA86A85F, 0xBEBFE988, 0x64E4C3FE, 0x9DBC8057, 0xF0F7C086,
      0x60787BF8, 0x6003604D, 0xD1FD8346, 0xF6381FB0, 0x7745AE04, 0xD736FCCC,
      0x83426B33, 0xF01EAB71, 0xB0804187, 0x3C005E5F, 0x77A057BE, 0xBDE8AE24,
      0x55464299, 0xBF582E61, 0x4E58F48F, 0xF2DDFDA2, 0xF474EF38, 0x8789BDC2,
      0x5366F9C3, 0xC8B38E74, 0xB475F255, 0x46FCD9B9, 0x7AEB2661, 0x8B1DDF84,
      0x846A0E79, 0x915F95E2, 0x466E598E, 0x20B45770, 0x8CD55591, 0xC902DE4C,
      0xB90BACE1, 0xBB8205D0, 0x11A86248, 0x7574A99E, 0xB77F19B6, 0xE0A9DC09,
      0x662D09A1, 0xC4324633, 0xE85A1F02, 0x09F0BE8C, 0x4A99A025, 0x1D6EFE10,
      0x1AB93D1D, 0x0BA5A4DF, 0xA186F20F, 0x2868F169, 0xDCB7DA83, 0x573906FE,
      0xA1E2CE9B, 0x4FCD7F52, 0x50115E01, 0xA70683FA, 0xA002B5C4, 0x0DE6D027,
      0x9AF88C27, 0x773F8641, 0xC3604C06, 0x61A806B5, 0xF0177A28, 0xC0F586E0,
      0x006058AA, 0x30DC7D62, 0x11E69ED7, 0x2338EA63, 0x53C2DD94, 0xC2C21634,
      0xBBCBEE56, 0x90BCB6DE, 0xEBFC7DA1, 0xCE591D76, 0x6F05E409, 0x4B7C0188,
      0x39720A3D, 0x7C927C24, 0x86E3725F, 0x724D9DB9, 0x1AC15BB4, 0xD39EB8FC,
      0xED545578, 0x08FCA5B5, 0xD83D7CD3, 0x4DAD0FC4, 0x1E50EF5E, 0xB161E6F8,
      0xA28514D9, 0x6C51133C, 0x6FD5C7E7, 0x56E14EC4, 0x362ABFCE, 0xDDC6C837,
      0xD79A3234, 0x92638212, 0x670EFA8E, 0x406000E0 };

__constant__ const u32 ks3[256] =
    { 0x3A39CE37, 0xD3FAF5CF, 0xABC27737, 0x5AC52D1B, 0x5CB0679E, 0x4FA33742,
      0xD3822740, 0x99BC9BBE, 0xD5118E9D, 0xBF0F7315, 0xD62D1C7E, 0xC700C47B,
      0xB78C1B6B, 0x21A19045, 0xB26EB1BE, 0x6A366EB4, 0x5748AB2F, 0xBC946E79,
      0xC6A376D2, 0x6549C2C8, 0x530FF8EE, 0x468DDE7D, 0xD5730A1D, 0x4CD04DC6,
      0x2939BBDB, 0xA9BA4650, 0xAC9526E8, 0xBE5EE304, 0xA1FAD5F0, 0x6A2D519A,
      0x63EF8CE2, 0x9A86EE22, 0xC089C2B8, 0x43242EF6, 0xA51E03AA, 0x9CF2D0A4,
      0x83C061BA, 0x9BE96A4D, 0x8FE51550, 0xBA645BD6, 0x2826A2F9, 0xA73A3AE1,
      0x4BA99586, 0xEF5562E9, 0xC72FEFD3, 0xF752F7DA, 0x3F046F69, 0x77FA0A59,
      0x80E4A915, 0x87B08601, 0x9B09E6AD, 0x3B3EE593, 0xE990FD5A, 0x9E34D797,
      0x2CF0B7D9, 0x022B8B51, 0x96D5AC3A, 0x017DA67D, 0xD1CF3ED6, 0x7C7D2D28,
      0x1F9F25CF, 0xADF2B89B, 0x5AD6B472, 0x5A88F54C, 0xE029AC71, 0xE019A5E6,
      0x47B0ACFD, 0xED93FA9B, 0xE8D3C48D, 0x283B57CC, 0xF8D56629, 0x79132E28,
      0x785F0191, 0xED756055, 0xF7960E44, 0xE3D35E8C, 0x15056DD4, 0x88F46DBA,
      0x03A16125, 0x0564F0BD, 0xC3EB9E15, 0x3C9057A2, 0x97271AEC, 0xA93A072A,
      0x1B3F6D9B, 0x1E6321F5, 0xF59C66FB, 0x26DCF319, 0x7533D928, 0xB155FDF5,
      0x03563482, 0x8ABA3CBB, 0x28517711, 0xC20AD9F8, 0xABCC5167, 0xCCAD925F,
      0x4DE81751, 0x3830DC8E, 0x379D5862, 0x9320F991, 0xEA7A90C2, 0xFB3E7BCE,
      0x5121CE64, 0x774FBE32, 0xA8B6E37E, 0xC3293D46, 0x48DE5369, 0x6413E680,
      0xA2AE0810, 0xDD6DB224, 0x69852DFD, 0x09072166, 0xB39A460A, 0x6445C0DD,
      0x586CDECF, 0x1C20C8AE, 0x5BBEF7DD, 0x1B588D40, 0xCCD2017F, 0x6BB4E3BB,
      0xDDA26A7E, 0x3A59FF45, 0x3E350A44, 0xBCB4CDD5, 0x72EACEA8, 0xFA6484BB,
      0x8D6612AE, 0xBF3C6F47, 0xD29BE463, 0x542F5D9E, 0xAEC2771B, 0xF64E6370,
      0x740E0D8D, 0xE75B1357, 0xF8721671, 0xAF537D5D, 0x4040CB08, 0x4EB4E2CC,
      0x34D2466A, 0x0115AF84, 0xE1B00428, 0x95983A1D, 0x06B89FB4, 0xCE6EA048,
      0x6F3F3B82, 0x3520AB82, 0x011A1D4B, 0x277227F8, 0x611560B1, 0xE7933FDC,
      0xBB3A792B, 0x344525BD, 0xA08839E1, 0x51CE794B, 0x2F32C9B7, 0xA01FBAC9,
      0xE01CC87E, 0xBCC7D1F6, 0xCF0111C3, 0xA1E8AAC7, 0x1A908749, 0xD44FBD9A,
      0xD0DADECB, 0xD50ADA38, 0x0339C32A, 0xC6913667, 0x8DF9317C, 0xE0B12B4F,
      0xF79E59B7, 0x43F5BB3A, 0xF2D519FF, 0x27D9459C, 0xBF97222C, 0x15E6FC2A,
      0x0F91FC71, 0x9B941525, 0xFAE59361, 0xCEB69CEB, 0xC2A86459, 0x12BAA8D1,
      0xB6C1075E, 0xE3056A0C, 0x10D25065, 0xCB03A442, 0xE0EC6E0E, 0x1698DB3B,
      0x4C98A0BE, 0x3278E964, 0x9F1F9532, 0xE0D392DF, 0xD3A0342B, 0x8971F21E,
      0x1B0A7441, 0x4BA3348C, 0xC5BE7120, 0xC37632D8, 0xDF359F8D, 0x9B992F2E,
      0xE60B6F47, 0x0FE3F11D, 0xE54CDA54, 0x1EDAD891, 0xCE6279CF, 0xCD3E7E6F,
      0x1618B166, 0xFD2C1D05, 0x848FD2C5, 0xF6FB2299, 0xF523F357, 0xA6327623,
      0x93A83531, 0x56CCCD02, 0xACF08162, 0x5A75EBB5, 0x6E163697, 0x88D273CC,
      0xDE966292, 0x81B949D0, 0x4C50901B, 0x71C65614, 0xE6C6C7BD, 0x327A140A,
      0x45E1D006, 0xC3F27B9A, 0xC9AA53FD, 0x62A80F00, 0xBB25BFE2, 0x35BDD2F6,
      0x71126905, 0xB2040222, 0xB6CBCF7C, 0xCD769C2B, 0x53113EC0, 0x1640E3D3,
      0x38ABBD60, 0x2547ADF0, 0xBA38209C, 0xF746CE76, 0x77AFA1C5, 0x20756060,
      0x85CBFE4E, 0x8AE88DD8, 0x7AAAF9B0, 0x4CF9AA7E, 0x1948C25C, 0x02FB8A8C,
      0x01C36AE4, 0xD6EBE1F9, 0x90D4F869, 0xA65CDEA0, 0x3F09252D, 0xC208E69F,
      0xB74E6132, 0xCE77E25B, 0x578FDFE3, 0x3AC372E6 };

__constant__ const u32 ps[N_BLOWFISH_ROUNDS + 2] =
    { 0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0,
      0x082EFA98, 0xEC4E6C89, 0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
      0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917, 0x9216D5D9, 0x8979FB1B };

#define F(x)                                            \
  (((subkey.s0[((u8*)&x)[3]] + subkey.s1[((u8*)&x)[2]]) \
    ^ subkey.s2[((u8*)&x)[1]])                          \
   + subkey.s3[((u8*)&x)[0]])
#define R(l, r, i)    \
  do {                \
    l ^= subkey.p[i]; \
    r ^= F(l);        \
  } while (0)

__device__ void BlowfishCreateSubkey(
    BlowfishSubkey& subkey_out, u8* key, u32 key_len)
{
  u32 i;
  u32 j;
  u32 data;
  u32 left;
  u32 right;

  for (i = 0; i < N_BLOWFISH_ROUNDS + 2; i++) {
    subkey_out.p[i] = ps[i];
  }
  for (i = 0; i < 256; i++) {
    subkey_out.s0[i] = ks0[i];
    subkey_out.s1[i] = ks1[i];
    subkey_out.s2[i] = ks2[i];
    subkey_out.s3[i] = ks3[i];
  }

  for (i = j = 0; i < N_BLOWFISH_ROUNDS + 2; i++) {
    data = key[j] << 24 | (key[(j + 1) % key_len]) << 16
           | (key[(j + 2) % key_len]) << 8 | (key[(j + 3) % key_len]);
    subkey_out.p[i] ^= data;
    j = (j + 4) % key_len;
  }

  left = right = 0;
  for (i = 0; i < N_BLOWFISH_ROUNDS + 2; i += 2) {
    BlowfishEncrypt(subkey_out, &left, &right);
    subkey_out.p[i] = left;
    subkey_out.p[i + 1] = right;
  }
  // TODO: Put all in one loop?
  for (i = 0; i < 256; i += 2) {
    BlowfishEncrypt(subkey_out, &left, &right);
    subkey_out.s0[i] = left;
    subkey_out.s0[i + 1] = right;
  }
  for (i = 0; i < 256; i += 2) {
    BlowfishEncrypt(subkey_out, &left, &right);
    subkey_out.s1[i] = left;
    subkey_out.s1[i + 1] = right;
  }
  for (i = 0; i < 256; i += 2) {
    BlowfishEncrypt(subkey_out, &left, &right);
    subkey_out.s2[i] = left;
    subkey_out.s2[i + 1] = right;
  }
  for (i = 0; i < 256; i += 2) {
    BlowfishEncrypt(subkey_out, &left, &right);
    subkey_out.s3[i] = left;
    subkey_out.s3[i + 1] = right;
  }
}

__device__ void BlowfishEncrypt(
    BlowfishSubkey& subkey, u32* left_out, u32* right_out)
{
  u32 left(*left_out);
  u32 right(*right_out);

  R(left, right, 0);
  R(right, left, 1);
  R(left, right, 2);
  R(right, left, 3);
  R(left, right, 4);
  R(right, left, 5);
  R(left, right, 6);
  R(right, left, 7);
  R(left, right, 8);
  R(right, left, 9);
  R(left, right, 10);
  R(right, left, 11);
  R(left, right, 12);
  R(right, left, 13);
  R(left, right, 14);
  R(right, left, 15);

  left ^= subkey.p[N_BLOWFISH_ROUNDS];
  right ^= subkey.p[N_BLOWFISH_ROUNDS + 1];

  *left_out = right;
  *right_out = left;
}

// Not in use.
//__device__ void BlowfishDecrypt(BlowfishSubkey& subkey, u32* left_out, u32*
//right_out)
//{
//  u32 left(*left_out);
//  u32 right(*right_out);
//
//  R(left, right, 17); R(right, left, 16); R(left, right, 15); R(right, left,
//  14);
//  R(left, right, 13); R(right, left, 12); R(left, right, 11); R(right, left,
//  10);
//  R(left, right, 9); R(right, left, 8); R(left, right, 7); R(right, left, 6);
//  R(left, right, 5); R(right, left, 4); R(left, right, 3); R(right, left, 2);
//
//  left ^= subkey.p[1];
//  right ^= subkey.p[0];
//
//  *left_out = right;
//  *right_out = left;
//}

__device__ void BlowfishDecryptBufferCFB(
    BlowfishSubkey& subkey, u32* buf_out, u32* buf_in, u32 n_blocks, u32* iv)
{
  u32 d1(iv[0]);
  u32 d2(iv[1]);

  for (int i(0); i < n_blocks * 2; i += 2) {
    BlowfishEncrypt(subkey, &d1, &d2);

    u32 x1(SwapEndian32(buf_in[i]));
    u32 x2(SwapEndian32(buf_in[i + 1]));

    buf_out[i] = SwapEndian32(x1 ^ d1);
    buf_out[i + 1] = SwapEndian32(x2 ^ d2);

    d1 = x1;
    d2 = x2;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Util, device

// Inner loop: YES
__device__ u32 SwapEndian32(u32 x)
{
  return __byte_perm(x, 0, 0x0123);
}

__device__ void SwapEndian32Ptr(u32* x)
{
  *x = __byte_perm(*x, 0, 0x0123);
}

// Inner loop: NO
__device__ u32 Min(u32 a, u32 b)
{
  return a < b ? a : b;
}

///////////////////////////////////////////////////////////////////////////////
// Util, host

// Inner loop: NO
__host__ void HostSwapEndian32Ptr(u32* x)
{
  u32 a(*x);
  *x =
      (a << 24) | (a >> 24) | ((a & 0x0000ff00) << 8) | ((a & 0x00ff0000) >> 8);
}

// Inner loop: NO
u32v u8vToLittleEndianu32v(u8v& in)
{
  u32v out;
  for (u8v::iterator iter(in.begin()); iter != in.end(); iter += 4) {
    u32* in32(reinterpret_cast<u32*>(&*iter));
    HostSwapEndian32Ptr(in32);
    out.push_back(*in32);
  }
  return out;
}
