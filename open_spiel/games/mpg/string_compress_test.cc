#include "string_compress.h"

#include <iostream>
#include "open_spiel/spiel.h"

void test_compress(const std::string& s)
{
    std::cout << "Testing compression on string: " << s << std::endl;
    std::string cipher_text= zlib_compress(s);
    std::string plain_text= zlib_decompress(cipher_text);
    SPIEL_CHECK_TRUE(s==plain_text);
    std::cout << "Compression test passed!" << std::endl;

}

void test_base64(const std::string& s)
{
    std::cout << "Testing base64 on string: " << s << std::endl;
    std::string cipher_text=base64_encode(s);
    std::string plain_text=base64_decode(cipher_text);
    SPIEL_CHECK_TRUE(s==plain_text);
    std::cout << "Base64 test passed!" << std::endl;

}

void test_compress_base64(const std::string &A)
{
    std::cout << "Testing compression and base64 on string: " << A << std::endl;
    std::string cipher_text=base64_encode(zlib_compress(A));
    std::string plain_text= zlib_decompress(base64_decode(cipher_text));
    SPIEL_CHECK_TRUE(A==plain_text);
    std::cout << "Compression and base64 test passed!" << std::endl;
}

bool is_base64(const std::string &A)
{
    for(int i=0;i<A.size();i++)
    {
        if(A[i] >= 'A' && A[i] <= 'Z')
            continue;
        if(A[i] >= 'a' && A[i] <= 'z')
            continue;
        if(A[i] >= '0' && A[i] <= '9')
            continue;
        if(A[i]=='+' || A[i]=='/')
            continue;
        if(A[i]=='=')
        {
            for (; i < A.size(); i++) if (A[i] != '=')
                return false;
            return true;
        }
        return false;
    }
    if(A.size()%4!=0)
        return false;
    return true;
}

void test_compressed_base64_encode(const std::string &A)
{
    std::cout << "Testing base64 validity on string: " << A << std::endl;
    auto Z= zlib_compress(A);
    std::string cipher_text=base64_encode(Z);
    std::cout << "Base64 encoded string: " << cipher_text << std::endl;
    auto D=base64_decode(cipher_text);
    SPIEL_CHECK_TRUE(is_base64(cipher_text));
    SPIEL_CHECK_TRUE(D==Z);
}

int main()
{


    std::string latin_gibberish=R"(Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean gravida laoreet odio, sodales dignissim sem mattis in. Duis non purus diam. Nam ullamcorper turpis tincidunt mattis imperdiet. Sed at venenatis mi, non dapibus orci. Vivamus at scelerisque metus. Donec cursus maximus justo, id sagittis quam feugiat interdum. Nunc neque elit, rhoncus id leo eget, efficitur tincidunt metus. Donec non fermentum nisi. Donec feugiat congue lorem ut pellentesque. Nunc gravida, est nec interdum eleifend, diam lectus elementum neque, sed aliquam velit ipsum ut nisi. Proin sodales ac lacus quis finibus. In vehicula eros dictum massa varius ornare. Donec eget tincidunt mi. Nunc a molestie leo. Pellentesque dapibus quam neque, ut convallis dolor sodales ut. Curabitur vitae dolor convallis, iaculis nulla at, aliquam ex.

Quisque posuere felis at velit eleifend posuere. Aliquam ultrices scelerisque massa. Curabitur tempor, lectus a iaculis hendrerit, arcu lectus posuere neque, vitae luctus arcu lacus eu orci. Quisque suscipit quis felis sed imperdiet. Sed efficitur viverra eros et dictum. Nam metus massa, sagittis non turpis ac, elementum suscipit mauris. Vestibulum non nulla quis nisl euismod suscipit. Ut efficitur pulvinar nulla, ut imperdiet sapien sagittis sed. Nullam tincidunt faucibus odio, vitae condimentum orci faucibus id. Aliquam eget ipsum suscipit, laoreet lorem vitae, viverra nunc.

Praesent ac nisi eget magna aliquet pulvinar. Donec auctor felis in dignissim dapibus. Nullam vel turpis ut leo pulvinar porta at sed mauris. Fusce venenatis ac nulla pulvinar viverra. Aliquam pharetra tincidunt justo ut tincidunt. Nullam dapibus turpis nec tortor laoreet, quis iaculis mi interdum. Suspendisse viverra eget ligula eget semper. Vestibulum ac erat condimentum, fermentum eros in, ultrices risus. Vivamus eget elit porta, tempus elit et, accumsan ligula. Sed auctor luctus finibus. Ut vitae eleifend nisi. Curabitur sit amet dignissim turpis.

In metus justo, maximus vel condimentum id, scelerisque nec arcu. In sit amet aliquam libero, eget rutrum quam. Vivamus gravida neque vitae imperdiet ultrices. Nunc purus quam, porta ac est eu, hendrerit rhoncus felis. Vestibulum vitae consequat elit, et venenatis urna. Nunc malesuada leo sed felis fermentum mattis. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Pellentesque ligula velit, tincidunt et hendrerit eget, finibus ac massa. In at metus tortor. Quisque et leo fermentum, efficitur est vitae, bibendum mi. Sed ornare sollicitudin dolor, eu auctor metus scelerisque vel. Phasellus molestie, eros malesuada ultricies hendrerit, felis ligula aliquet est, at maximus velit tellus et quam. Curabitur ut ex quam.

Praesent convallis lorem non rutrum iaculis. Cras vel scelerisque lacus, ac semper augue. Integer dignissim placerat libero, eget finibus urna interdum ut. Aliquam nibh enim, auctor sed massa eu, luctus finibus est. Praesent neque est, tempor nec turpis id, facilisis facilisis lectus. Nam pulvinar elit interdum quam consectetur commodo. Nam nec massa in justo vestibulum dapibus. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas euismod ex ac diam blandit dignissim. Donec ut nunc justo. Cras rutrum non tellus lacinia tincidunt. Curabitur scelerisque a erat vel cursus.)";

    std::vector<std::string> strings={"Hola","Hello World","Mean Payoff Game",latin_gibberish};
    std::cout << "Testing string compression" << std::endl;
    for(auto &A:strings)
        test_compress(A);
    std::cout << "***" << std::endl;
    std::cout << "Testing string base64" << std::endl;
    for(auto &A:strings)
        test_base64(A);
    std::cout << "***" << std::endl;

    std::cout << "Testing base64 validity on compressed strings" << std::endl;
    for(auto &A:strings)
        test_compressed_base64_encode(A);
    std::cout << "***" << std::endl;

    std::cout << "Testing string compression and base64" << std::endl;
    for(auto &A:strings)
        test_compress_base64(A);
    std::cout << "***" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}