//
// Created by ramizouari on 07/06/23.
//

#ifndef OPEN_SPIEL_STRING_COMPRESS_H
#define OPEN_SPIEL_STRING_COMPRESS_H
#include <iostream>

std::string zlib_compress(const std::string &data);
std::string zlib_decompress(const std::string &cipher_text);
std::string bzip_compress(const std::string &data);
std::string bzip_decompress(const std::string &cipher_text);
std::string base64_encode(const std::string &data);
std::string base64_decode(const std::string &cipher_text);
#endif //OPEN_SPIEL_STRING_COMPRESS_H
