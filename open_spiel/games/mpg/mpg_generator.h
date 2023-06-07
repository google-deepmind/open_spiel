//
// Created by ramizouari on 05/06/23.
//

#ifndef OPEN_SPIEL_MPG_GENERATOR_H
#define OPEN_SPIEL_MPG_GENERATOR_H
#include "mpg.h"
namespace open_spiel::mpg
{
    /**
     * @brief Choose a random subset of k elements from a set v
     * @tparam T type of elements in v
     * @tparam Compare type of comparison function for elements in v
     * @tparam Allocator type of allocator for elements in v
     * @tparam RNG type of random number generator
     * @param v set of elements to choose from
     * @param k number of elements to choose
     * @param rng random number generator
     * @return a vector of k elements from v
     * */
    template<typename T,typename Compare,typename Allocator,typename RNG>
    std::set<T,Allocator> choose_set(const std::vector<T,Allocator> &v, int k, RNG && rng)
    {
        std::uniform_int_distribution<size_t> dist(0,v.size()-1);
        if (k>v.size())
            throw std::invalid_argument("k must be less than or equal to v.size()");
        std::set<size_t> result;
        while(result.size()<k)
        {
            auto j=dist(rng);
            result.insert(v[j]);
        }
        return result;
    }

    template<typename T,typename Compare,typename Allocator,typename RNG>
    std::set<T,Allocator> choose_set(const std::set<T,Compare,Allocator> &v, int k, RNG && rng)
    {
        std::vector<T,Allocator> v2(v.begin(),v.end());
        return choose_set(v2,k,std::forward<RNG>(rng));
    }

    /**
    * @brief Choose a random subset of k elements from a set v
    * @tparam T type of elements in v
    * @tparam Compare type of comparison function for elements in v
    * @tparam Allocator type of allocator for elements in v
    * @tparam RNG type of random number generator
    * @param v set of elements to choose from
    * @param k number of elements to choose
    * @param rng random number generator
    * @return a vector of k elements from v
    * @note The elements in the returned multiset can be repeated
    * */
    template<typename T,typename Compare,typename Allocator,typename RNG>
    std::multiset<T,Allocator> choose_multiset(const std::vector<T,Allocator> &v, int k, RNG && rng)
    {
        std::uniform_int_distribution<size_t> dist(0,v.size()-1);
        if (k>v.size())
            throw std::invalid_argument("k must be less than or equal to v.size() for distinct=true");
          std::multiset<T,Compare,Allocator> result;
        for(int i=0;i<k;i++)
        {
            auto j=dist(rng);
            result.insert(v[j]);
        }
        return result;
    }

    template<typename T,typename Compare,typename Allocator,typename RNG>
    std::multiset<T,Allocator> choose_multiset(const std::multiset<T,Compare,Allocator> &v, int k, RNG && rng)
    {
        std::vector<T,Allocator> v2(v.begin(),v.end());
        return choose_multiset(v2,k,std::forward<RNG>(rng));
    }

    /**
     * @brief Choose a random subset of k elements from a set v
     * @tparam T type of elements in v
     * @tparam Compare type of comparison function for elements in v
     * @tparam Allocator type of allocator for elements in v
     * @tparam RNG type of random number generator
     * @param v set of elements to choose from
     * @param k number of elements to choose
     * @param rng random number generator
     * @return a vector of k elements from v
     * */
    template<typename T,typename Hash,typename Equality,typename Allocator,typename RNG>
    std::unordered_set<T,Hash,Equality,Allocator> choose_set(const std::vector<T,Allocator> &v, int k, RNG && rng)
    {
        std::uniform_int_distribution<size_t> dist(0,v.size()-1);
        if (k>v.size())
            throw std::invalid_argument("k must be less than or equal to v.size()");
        std::unordered_set<T,Hash,Equality,Allocator> result;
        while(result.size()<k)
        {
            auto j=dist(rng);
            result.insert(v[j]);
        }
        return result;
    }

    template<typename T,typename Hash,typename Equality,typename Allocator,typename RNG>
    std::unordered_set<T,Hash,Equality,Allocator> choose_set(const std::unordered_set<T,Hash,Equality,Allocator> &v, int k, RNG && rng)
    {
        std::vector<T,Allocator> v2(v.begin(),v.end());
        return choose_set(v2,k,std::forward<RNG>(rng));
    }

    /**
    * @brief Choose a random subset of k elements from a set v
    * @tparam T type of elements in v
    * @tparam Compare type of comparison function for elements in v
    * @tparam Allocator type of allocator for elements in v
    * @tparam RNG type of random number generator
    * @param v set of elements to choose from
    * @param k number of elements to choose
    * @param rng random number generator
    * @return a vector of k elements from v
    * @note The elements in the returned multiset can be repeated
    * */
    template<typename T,typename Hash,typename Equality,typename Allocator,typename RNG>
    std::unordered_multiset<T,Hash,Equality,Allocator> choose_multiset(const std::vector<T,Allocator> &v, int k, RNG && rng)
    {
        std::uniform_int_distribution<size_t> dist(0,v.size()-1);
        if (k>v.size())
            throw std::invalid_argument("k must be less than or equal to v.size() for distinct=true");
        std::unordered_multiset<T,Hash,Equality,Allocator> result;
        for(int i=0;i<k;i++)
        {
            auto j=dist(rng);
            result.insert(v[j]);
        }
        return result;
    }

    template<typename T,typename Hash,typename Equality,typename Allocator,typename RNG>
    std::unordered_multiset<T,Hash,Equality,Allocator> choose_multiset(const std::unordered_multiset<T,Hash,Equality,Allocator> &v, int k, RNG && rng)
    {
        std::vector<T,Allocator> v2(v.begin(),v.end());
        return choose_multiset(v2,k,std::forward<RNG>(rng));
    }


    namespace choose_parameters
    {
            inline constexpr float threshold=0.5;
    }

    /**
     * @brief Choose a random subset of k elements from a set v
     * @tparam T type of elements in v
     * @tparam Allocator type of allocator for elements in v
     * @tparam RNG type of random number generator
     * @param v vector of elements to choose from
     * @param k number of elements to choose
     * @param rng random number generator
     * @param distinct if true, the elements in the returned vector are distinct (no index is repeated)
     * @return a vector of k elements from v
     * */
    template<typename T,typename Allocator,typename RNG>
    std::vector<T,Allocator> choose(const std::vector<T,Allocator> &v,int k, RNG && rng,bool distinct=true)
    {

        std::uniform_int_distribution<size_t> dist(0,v.size()-1);
        if(distinct)
        {
            if(k>v.size())
                throw std::invalid_argument("k must be less than or equal to v.size() for distinct=true");
            std::vector<T,Allocator> result;
            std::unordered_set<size_t> v_set;
            if(k<= choose_parameters::threshold * v.size()) while(v_set.size()<k)
            {
                auto j=dist(rng);
                v_set.insert(j);
            }
            else
            {
                for(int i=0;i<v.size();i++)
                    v_set.insert(i);
                while(v_set.size() > k)
                {
                    auto j=dist(rng);
                    v_set.erase(j);
                }
            }
            for(auto i:v_set)
                result.push_back(v[i]);
            return result;
        }
        else
        {
            std::vector<T,Allocator> result;
            for(int i=0;i<k;i++)
            {
                auto j=dist(rng);
                result.push_back(v[j]);
            }
            return result;
        }
    }

    std::vector<NodeType> choose(NodeType n,int k, std::mt19937_64 & rng,bool distinct=true);

    class GraphGenerator
    {
    public:
        virtual GraphType operator()()=0;
        virtual ~GraphGenerator()=default;
    };

    class GnpGenerator:public GraphGenerator
    {
        std::uint64_t n;
        double p;
        std::mt19937_64 rng;
    public:
        GnpGenerator(std::uint64_t n, double p, std::uint64_t seed = 0);
        GraphType operator()() override;
    };

    class SinklessGnpGenerator: public GraphGenerator
    {
        std::uint64_t n;
        double p;
        std::mt19937_64 rng;
    public:
        SinklessGnpGenerator(std::uint64_t n, double p, std::uint64_t seed = 0);
        GraphType operator()() override;
    };


    class WeightGenerator
    {
    public:
        virtual ~WeightGenerator()=default;
        virtual WeightType operator()()=0;
    };

    class UniformWeightGenerator:public WeightGenerator, public std::uniform_real_distribution<WeightType>
    {
        std::mt19937_64 rng;
    public:
        UniformWeightGenerator(WeightType a, WeightType b, std::uint64_t seed = 0);
        WeightType operator()() override;
    };

    class NormalWeightGenerator:public WeightGenerator, public std::normal_distribution<WeightType>
    {
        std::mt19937_64 rng;
    public:
        NormalWeightGenerator(WeightType mean, WeightType std, std::uint64_t seed = 0);
        WeightType operator()() override;
    };


    class DiscreteUniformWeightGenerator:public WeightGenerator, public std::uniform_int_distribution<size_t>
    {
        std::mt19937_64 rng;
    public:
        DiscreteUniformWeightGenerator(size_t a, size_t b, std::uint64_t seed = 0);
        WeightType operator()() override;
    };


    class WeightedGraphGenerator
    {
        std::shared_ptr<GraphGenerator> graph_generator;
        std::shared_ptr<WeightGenerator> weight_generator;
    public:
        WeightedGraphGenerator(std::shared_ptr<GraphGenerator> graph_generator, std::shared_ptr<WeightGenerator> weight_generator);
        WeightedGraphType operator()();

    };

    class MetaFactory
    {
    public:
        virtual ~MetaFactory() = default;
        virtual std::shared_ptr<const Game> CreateGame(const GameParameters& params)  = 0;
    };

    class ParserMetaFactory : public MetaFactory
    {
    public:
        std::shared_ptr<const Game> CreateGame(const GameParameters& params) override;
    };
    extern std::unique_ptr<MetaFactory> metaFactory;

    class EnvironmentFactory
    {
        public:
        virtual ~EnvironmentFactory()=default;
        virtual std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame &metaGame)=0;
    };


    class GeneratorEnvironmentFactory : public EnvironmentFactory
    {
        std::shared_ptr<WeightedGraphGenerator> weighted_graph_generator;
        std::mt19937_64 rng;
    public:
        explicit GeneratorEnvironmentFactory(std::shared_ptr<WeightedGraphGenerator> weighted_graph_generator, std::uint64_t seed=0);
        [[nodiscard]] std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame &metaGame)  override;
    };

    class UniformGnpEnvironmentFactory : public GeneratorEnvironmentFactory
    {

    public:
        UniformGnpEnvironmentFactory(NodeType n, WeightType p, WeightType a, WeightType b, std::uint64_t seed = 0);
    };

    class DeterministicEnvironmentFactory : public EnvironmentFactory
    {
    public:
        ~DeterministicEnvironmentFactory() override = default;
        std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame& metaGame)  override = 0;
    };

    class ExampleEnvironmentFactory : public DeterministicEnvironmentFactory
    {
    public:
        ~ExampleEnvironmentFactory() override = default;
        std::shared_ptr<Environment> NewEnvironment(const MPGMetaGame& metaGame) override;
    };
}

#endif //OPEN_SPIEL_MPG_GENERATOR_H
