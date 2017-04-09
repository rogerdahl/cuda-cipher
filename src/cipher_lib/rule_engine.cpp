#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

const char* rules_path("./rules_test.txt");
const char* passwords_path("./words_short_selections.txt");

#include "passwords.h"

string trim(string& str);

int test_main()
{
  Passwords p(1000, 20);
  bool success(p.OpenPasswordFile(string(passwords_path)));
  if (!success) {
    cerr << "Couldn't open \"" << passwords_path << "\"" << endl;
    exit(1);
  }

  while (true) {
    int n = p.GetPasswords();
    if (!n) {
      break;
    }
    for (int i(0); i < n; ++i) {
      cout << p.GetPasswordBuf() + i * 20 << "\n";
    }
  }

  exit(0);

  fstream i(rules_path);

  if (!i.good()) {
    cerr << "Couldn't open \"" << rules_path << "\"";
    exit(1);
  }

  string line;
  while (i.good()) {
    getline(i, line);
    line = trim(line);

    if (line[0] == '#' || line == "") {
      continue;
    }

    cout << line << endl;
  }

  return 0;
}

string trim(string& s)
{
  s.erase(0, s.find_first_not_of(' '));
  s.erase(s.find_last_not_of(' ') + 1);
  return s;
}

void ProcessCommands(string& s)
{
  while (s != "") {
    switch (s[0]) {
    // Nothing 	: 	do nothing 	: 	p@ssW0rd 	p@ssW0rd
    case ':':
      break;

    // Lowercase 	l 	Lowercase all letters 	l 	p@ssW0rd 	p@ssw0rd
    case 'l':
      // run(lc);
      // s = s.sub(1);
      break;

    // Uppercase 	u 	Uppercase all letters 	u 	p@ssW0rd 	P@SSW0RD
    case 'u':
      break;

    // Capitalize 	c 	Capitalize the first letter 	c 	p@ssW0rd 	P@ssW0rd
    case 'c':
      break;

    // Invert Capitalize 	C 	Lowercase first found character, uppercase the rest
    // C 	p@ssW0rd 	p@SSW0RD
    case 'C':
      break;

    // Toggle Case 	t 	Toggle the case of all characters in word. 	t 	p@ssW0rd
    // P@SSw0RD
    case 't':
      break;

    // Toggle @ 	TN 	Toggle the case of characters at position N 	T3 	p@ssW0rd
    // p@sSW0rd
    case 'T':
      break;

    // Reverse 	r 	Reverse the entire word 	r 	p@ssW0rd 	dr0Wss@p
    case 'r':
      break;

    // Duplicate 	d 	Duplicate entire word 	d 	p@ssW0rd 	p@ssW0rdp@ssW0rd
    case 'd':
      break;

    // Reflect 	f 	Duplicate word reversed 	f 	p@ssW0rd 	p@ssW0rddr0Wss@p
    case 'f':
      break;

    // Rotate Left 	{ 	Rotates the word left. 	{ 	p@ssW0rd 	@ssW0rdp
    case '{':
      break;

    // Rotate Right 	} 	Rotates the word right 	} 	p@ssW0rd 	dp@ssW0r
    case '}':
      break;

    // Append Character 	$ 	Append character to end 	$1 	p@ssW0rd 	p@ssW0rd1
    case '$':
      break;

    // Prepend Character 	^ 	Prepend character to front 	^1 	p@ssW0rd 1p@ssW0rd
    case '^':
      break;

    // Truncate left 	[ 	Deletes first character 	[ 	p@ssW0rd 	@ssW0rd
    case '[':
      break;

    // Trucate right 	] 	Deletes last character 	] 	p@ssW0rd 	p@assW0r
    case ']':
      break;

    // Delete @ N 	DN 	Deletes character at position N 	D3 	p@ssW0rd 	p@sW0rd
    // *
    case 'D':
      break;

    // Delete range 	xNM 	Deletes M characters, starting at position N 	x02
    // p@ssW0rd 	ssW0rd 	*
    case 'x':
      break;

    // Insert @ N 	iNX 	Inserts character X at position N 	i4! 	p@ssW0rd
    // p@ss!W0rd 	*
    case 'i':
      break;

    // Overwrite @ N 	oNX 	Overwrites character at postion N with X 	o3$
    // p@ssW0rd 	p@s$W0rd 	*
    case 'o':
      break;

    // Truncate @ N 	'N 	Truncate word at position N 	'6 	p@ssW0rd 	p@ssW0
    case '\'':
      break;

    // Replace 	sXY 	Replace all instances of X with Y 	ss$ 	p@ssW0rd
    // p@$$W0rd
    case 's':
      break;

    // Purge 	@X 	Purge all instances of X 	@s 	p@ssW0rd 	p@W0rd 	+
    case '@':
      break;

    // Duplicate first N 	z 	Duplicates first character N times 	z2 	p@ssW0rd
    // ppp@ssW0rd
    case 'z':
      break;

    // Duplicate last N 	Z 	Duplicates last character N times 	Z2 	p@ssW0rd
    // p@ssW0rddd
    case 'Z':
      break;

    // Duplicate all 	q 	Duplicate every character 	q 	p@ssW0rd
    // pp@@ssssWW00rrdd
    case 'q':
      break;

    // Duplicate word 	pN 	Duplicate entire word N times 	p3 	Pass
    // PassPassPass
    case 'p':
      break;

    // Swap front 	k 	Swaps first two characters 	k 	p@ssW0rd 	@pssW0rd
    case 'k':
      break;

    // Swap back 	K 	Swaps last two characters 	K 	p@ssW0rd 	p@ssW0dr
    case 'K':
      break;

    // Swap @ N 	*XY 	Swaps character X with Y 	*34 	p@ssW0rd 	p@sWs0rd 	*
    case '*':
      break;

    // Bitwise shift left 	LN 	Bitwise shift left character @ N 	L2 	p@ssW0rd
    // p@æsW0rd 	*
    case 'L':
      break;

    // Bitwise shift right 	RN 	Bitwise shift right character @ N 	R2 	p@ssW0rd
    // p@9sW0rd 	*
    case 'R':
      break;

    // Ascii increment 	+N 	Increment character @ N by 1 ascii value 	+2
    // p@ssW0rd 	p@tsW0rd 	*
    case '+':
      break;

    // Ascii decrement 	-N 	Decrement character @ N by 1 ascii value 	-2
    // p@ssW0rd 	p?ssW0rd 	*
    case '-':
      break;

    // Replace N + 1 	.N 	Replaces character @ N with value at @ N plus 1 	.1
    // p@ssW0rd 	psssW0rd 	*
    case '.':
      break;

    // Replace N - 1 	,N 	Replaces character @ N with value at @ N minus 1 	,1
    // p@ssW0rd 	ppssW0rd 	*
    case ',':
      break;

    // Duplicate block front 	yN 	Duplicates first N characters 	y2 	p@ssW0rd
    // p@p@ssW0rd 	*
    case 'y':
      break;

    // Duplicate block back 	YN 	Duplicates last N characters 	Y2 	p@ssW0rd
    // p@ssW0rdrd 	*
    case 'Y':
      break;

    // Title 	E 	Upper case the first letter and every letter after a space 	E
    // p@ssW0rd w0rld 	P@ssw0rd W0rld
    case 'E':
      break;

      //  * Indicates that N starts at 0. For character positions other than 0-9
      //  use A-Z (A=11)
    }
  }
}
