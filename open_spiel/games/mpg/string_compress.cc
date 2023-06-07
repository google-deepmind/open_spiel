//
// Created by ramizouari on 07/06/23.
//
#include "string_compress.h"
#include "boost/beast/core/detail/base64.hpp"
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

#include <sstream>


std::string zlib_compress(const std::string &data)
{
    boost::iostreams::filtering_streambuf<boost::iostreams::output> output_stream;
    output_stream.push(boost::iostreams::zlib_compressor());
    std::stringstream string_stream;
    output_stream.push(string_stream);
    boost::iostreams::copy(boost::iostreams::basic_array_source<char>(data.c_str(),
                                                                      data.size()), output_stream);
    return string_stream.str();
}

std::string zlib_decompress(const std::string &cipher_text) {
    std::stringstream string_stream;
    string_stream << cipher_text;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> input_stream;
    input_stream.push(boost::iostreams::zlib_decompressor());

    input_stream.push(string_stream);
    std::stringstream unpacked_text;
    boost::iostreams::copy(input_stream, unpacked_text);
    return unpacked_text.str();
}

std::string bzip_compress(const std::string &data)
{
    boost::iostreams::filtering_streambuf<boost::iostreams::output> output_stream;
    output_stream.push(boost::iostreams::zlib_compressor());
    std::stringstream string_stream;
    output_stream.push(string_stream);
    boost::iostreams::copy(boost::iostreams::basic_array_source<char>(data.c_str(),
                                                                      data.size()), output_stream);
    return string_stream.str();
}

std::string bzip_decompress(const std::string &cipher_text) {
    std::stringstream string_stream;
    string_stream << cipher_text;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> input_stream;
    input_stream.push(boost::iostreams::zlib_decompressor());

    input_stream.push(string_stream);
    std::stringstream unpacked_text;
    boost::iostreams::copy(input_stream, unpacked_text);
    return unpacked_text.str();
}

std::string base64_encode(const std::string &data)
{
    auto raw_data=data.c_str();
    auto size=data.size();
    static std::string base64_chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";
    std::string result;
    result.reserve(4*((size+2)/3));
    for(int i=0;i<size;i+=3)
    {
        auto c1=static_cast<unsigned char>(raw_data[i]);
        auto c2=static_cast<unsigned char>((i+1<size)?raw_data[i+1]:0);
        auto c3=static_cast<unsigned char>((i+2<size)?raw_data[i+2]:0);
        auto e1=c1>>2;
        auto e2=((c1&0x3)<<4)|(c2>>4);
        auto e3=((c2&0xf)<<2)|(c3>>6);
        auto e4=c3&0x3f;
        result.push_back(base64_chars.at(e1));
        result.push_back(base64_chars.at(e2));
        result.push_back((i+1<size)?base64_chars[e3]:'=');
        result.push_back((i+2<size)?base64_chars[e4]:'=');
    }
    return result;
}

std::string base64_decode(const std::string &cipher_text)
{
    static std::string base64_chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";
    auto size=cipher_text.size();
    std::string result;
    result.reserve(3*(size/4));
    for(int i=0;i<size;i+=4)
    {
        auto c1=base64_chars.find(cipher_text[i]);
        auto c2=base64_chars.find(cipher_text[i+1]);
        auto c3=base64_chars.find(cipher_text[i+2]);
        auto c4=base64_chars.find(cipher_text[i+3]);
        auto e1=(c1<<2)|(c2>>4);
        auto e2=((c2&0xf)<<4)|(c3>>2);
        auto e3=((c3&0x3)<<6)|c4;
        result.push_back(e1);
        if(cipher_text[i+2]!='=')
            result.push_back(e2);
        if(cipher_text[i+3]!='=')
            result.push_back(e3);
    }
    return result;
}
