/*
   Based on the implementation https://github.com/ReneNyffenegger/cpp-base64
   and modified for the OpenSpiel framework, in accord to the original license.

   Copyright (C) 2004-2017, 2020, 2021 René Nyffenegger
   This source code is provided 'as-is', without any express or implied
   warranty. In no event will the author be held liable for any damages
   arising from the use of this software.
   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:
   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.
   3. This notice may not be removed or altered from any source distribution.

   René Nyffenegger rene.nyffenegger@adp-gmbh.ch

 */

#include "open_spiel/higc/base64.h"

namespace open_spiel {
namespace higc {


const char* base64_chars = {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "abcdefghijklmnopqrstuvwxyz"
                            "0123456789"
                            "+/"};
constexpr char trailing_char = '=';

void base64_encode(std::ostream& os, char const* buf, size_t len) {
  unsigned int pos = 0;
  while (pos < len) {
    os << base64_chars[(buf[pos + 0] & 0xfc) >> 2];
    if (pos + 1 < len) {
      os << base64_chars[((buf[pos + 0] & 0x03) << 4)
          + ((buf[pos + 1] & 0xf0) >> 4)];
      if (pos + 2 < len) {
        os << base64_chars[((buf[pos + 1] & 0x0f) << 2)
            + ((buf[pos + 2] & 0xc0) >> 6)];
        os << base64_chars[buf[pos + 2] & 0x3f];
      } else {
        os << base64_chars[(buf[pos + 1] & 0x0f) << 2];
        os << trailing_char;
      }
    } else {
      os << base64_chars[(buf[pos + 0] & 0x03) << 4];
      os << trailing_char;
      os << trailing_char;
    }
    pos += 3;
  }
}


// Return the position of chr within base64_encode()
unsigned int pos_of_char(const char chr) {
  if      (chr >= 'A' && chr <= 'Z') return chr - 'A';
  else if (chr >= 'a' && chr <= 'z') return chr - 'a' + ('Z' - 'A')               + 1;
  else if (chr >= '0' && chr <= '9') return chr - '0' + ('Z' - 'A') + ('z' - 'a') + 2;
  else if (chr == '+') return 62;
  else if (chr == '/') return 63;
  else throw std::runtime_error("Input is not valid base64-encoded data.");
}

std::string base64_decode(absl::string_view encoded_string) {
  if (encoded_string.empty()) return std::string();

  size_t length_of_string = encoded_string.length();
  size_t pos = 0;

  // The approximate length (bytes) of the decoded string might be one or
  // two bytes smaller, depending on the amount of trailing equal signs
  // in the encoded string. This approximation is needed to reserve
  // enough space in the string to be returned.
  size_t approx_length_of_decoded_string = length_of_string / 4 * 3;
  std::string ret;
  ret.reserve(approx_length_of_decoded_string);

  while (pos < length_of_string) {
    // Iterate over encoded input string in chunks. The size of all
    // chunks except the last one is 4 bytes.
    //
    // The last chunk might be padded with equal signs or dots
    // in order to make it 4 bytes in size as well, but this
    // is not required as per RFC 2045.
    //
    // All chunks except the last one produce three output bytes.
    //
    // The last chunk produces at least one and up to three bytes.
    size_t pos_of_char_1 = pos_of_char(encoded_string[pos + 1]);

    // Emit the first output byte that is produced in each chunk:
    ret.push_back(static_cast<std::string::value_type>(
                      ((pos_of_char(encoded_string[pos + 0])) << 2)
                          + ((pos_of_char_1 & 0x30) >> 4)
                  ));

    // Check for data that is not padded with equal signs
    // (which is allowed by RFC 2045)
    if ((pos + 2 < length_of_string) && encoded_string[pos + 2] != '=') {
      // Emit a chunk's second byte (which might not be produced in the last chunk).
      unsigned int pos_of_char_2 = pos_of_char(encoded_string[pos + 2]);
      ret.push_back(static_cast<std::string::value_type>(
                        ((pos_of_char_1 & 0x0f) << 4)
                            + ((pos_of_char_2 & 0x3c) >> 2)
                    ));

      if ((pos + 3 < length_of_string) && encoded_string[pos + 3] != '=') {
        // Emit a chunk's third byte (which might not be produced in the last chunk).
        ret.push_back(static_cast<std::string::value_type>(
                          ((pos_of_char_2 & 0x03) << 6)
                              + pos_of_char(encoded_string[pos + 3])
                      ));
      }
    }
    pos += 4;
  }

  return ret;
}


}  // namespace higc
}  // namespace open_spiel
