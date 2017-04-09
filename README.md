# OpenOffice Password Cracker

OpenOffice provides strong encryption for documents. This program uses CUDA capable GPUs to attempt to unlock the documents by trying a large number of generated passwords.


## OpenOffice encryption scheme

OpenOffice uses PBKDF2-HMAC-SHA1 to convert a user-supplied password to a key that is used for Blowfish decryption:

* Obtain a password of any size from the user
* Use SHA1 to create 20-byte hash of the password
* Seed a PRNG with current time
* For each file:
    * Use PRNG to generate random initialization vector (8 bytes) and salt (16 bytes)
    * Use 1024 rounds of PBKDF2-HMAC-SHA1 to generate a 128-bit encryption key from the init vector and salt
    * Use Blowfish CFB (cipher feedback) with the encryption key to encrypt the file


## Implementation

Implements a CUDA kernel that performs PBKDF2-HMAC-SHA1 and Blowfish CFB.

Input:

* `password`: The user supplied password. 1 - 60 bytes
* `salt`:
    * The salt makes it impossible to use a table of precomputed keys for known passwords
    * 1 - 60 bytes. Must be 64 bytes, zero padded
* `salt_len`: The size of the salt, minus the zero padding. Must be 1 - 60
* `key_len`: The size of the key to generate

Operation:

* OpenOffice runs HMAC-SHA1 only on SHA1 hashes
* Password check:
    * The SHA1 hash of the password is calculated
    * The SHA1 hash is run through 1024 rounds of HMAC-SHA1
    * Each round both takes and returns an SHA1 hash


## Optimization

* The functions have been optimized for handling the structures required for handling OpenOffice's encryption.

* By default, the compiler unrolls small loops with a known trip count, so I haven't used #pragma unroll.

* Each function is marked with Inner loop: YES/NO. Only functions in the inner loop are important for optimization, as they are part of PBKDF2 and run 1024 times per password, while the other functions only run one or a few times per password.

* Tried setting number of registers to 62 to increase occupancy, but the compiler created a non-working kernel (it doesn't find the password). Cuda 6.0 when compiling for either CC 2.0 or CC 5.0 and running on CC 5.0.

## PBKDF2 in pseudocode

```
function hmac (key, message)
  if (len(key) > blocksize) then
    key = hash(key) // keys longer than blocksize are shortened
  end if
  if (len(key) < blocksize) then
    key = key ∥ [0x00 * (blocksize - len(key))] // keys shorter than blocksize are zero-padded ('∥' is concatenation)
  end if

  o_key_pad = [0x5c * blocksize] ⊕ key // Where blocksize is that of the underlying hash function
  i_key_pad = [0x36 * blocksize] ⊕ key // Where ⊕ is exclusive or (XOR)

  return hash(o_key_pad ∥ hash(i_key_pad ∥ message)) // Where '∥' is concatenation
end function
```

## To do

Try: Split the kernel into two, one for PBKDF2 and one for Blowfish. I'm wondering if, when the same kernel does two completely different tasks, the two parts may interfere with caching for each other. Also, I don't think the compiler understands that PBKDF2 runs 1024 times while Blowfish runs only one time. So it might spill registers in PBKDF2 in order to make Blowfish run faster. Surprisingly, Blowfish currently takes almost the same amount of time as 1024 rounds of PBKDF2.

