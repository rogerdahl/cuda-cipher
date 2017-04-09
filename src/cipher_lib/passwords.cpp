#include <iostream>
#include <string>

#include <cstring>

#include "passwords.h"

using namespace std;

const int READ_BUF_SIZE(1024 * 1024);

Passwords::Passwords(int n_passwords, int max_password_len)
  : n_passwords_(n_passwords),
    max_password_len_(max_password_len),
    read_buf_pos_(0),
    n_bytes_in_buf_(0)
{
  password_buf_ = new char[n_passwords * max_password_len_];
  read_buf_ = new char[READ_BUF_SIZE];
}

Passwords::~Passwords()
{
  delete[] password_buf_;
  delete[] read_buf_;
}

bool Passwords::OpenPasswordFile(const string& path)
{
  ifs_.open(path.c_str());
  return ifs_.good();
}

int Passwords::GetPasswords()
{
  int i;
  for (i = 0; i < n_passwords_; ++i) {
    if (n_bytes_in_buf_ - read_buf_pos_ < max_password_len_) {
      ShiftAndRead();
      if (!n_bytes_in_buf_) {
        break;
      }
    }
    strcpy(
        password_buf_ + i * max_password_len_,
        read_buf_ + read_buf_pos_); // strcpy includes terminator.
    read_buf_pos_ += strlen(password_buf_ + i * max_password_len_)
                     + 1; // strlen does not count terminator
  }
  return i;
}

void Passwords::ShiftAndRead()
{
  memmove(
      read_buf_, read_buf_ + read_buf_pos_,
      static_cast<size_t>(n_bytes_in_buf_ - read_buf_pos_));
  n_bytes_in_buf_ -= read_buf_pos_;
  ifs_.read(read_buf_ + n_bytes_in_buf_, READ_BUF_SIZE - n_bytes_in_buf_);
  streamsize n_bytes_read(ifs_.gcount());
  ReplaceNewlineWithNull(read_buf_ + n_bytes_in_buf_, n_bytes_read);
  n_bytes_in_buf_ += n_bytes_read;
  read_buf_pos_ = 0;
}

void Passwords::ReplaceNewlineWithNull(char* start, streamsize len)
{
  for (int i(0); i < len; ++i) {
    if (start[i] == '\n') {
      start[i] = '\0';
    }
  }
}

char* Passwords::GetPasswordBuf()
{
  return password_buf_;
}
