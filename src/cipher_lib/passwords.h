#include <fstream>
#include <string>

class Passwords
{
  public:
  Passwords(int n_passwords, int max_password_len);
  ~Passwords();

  bool OpenPasswordFile(const std::string&);
  int GetPasswords();
  char* GetPasswordBuf();

  private:
  void ShiftAndRead();
  void ReplaceNewlineWithNull(char* start, std::streamsize len);

  std::ifstream ifs_;

  int n_passwords_;
  int max_password_len_;

  char* read_buf_;
  std::streamsize read_buf_pos_;
  std::streamsize n_bytes_in_buf_;

  char* password_buf_;
};
