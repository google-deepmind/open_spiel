//
// Created by ramizouari on 06/06/23.
//

#include "mpg_generator.h"

namespace open_spiel::mpg
{

    std::vector<NodeType> choose(NodeType n, int k, std::mt19937_64 &rng, bool distinct)
    {
        if(!distinct)
        {
            std::uniform_int_distribution<NodeType> dist(0,n-1);
            std::vector<NodeType> result(n);
            for(int i=0;i<k;i++)
            {
                auto j=dist(rng);
                result[i]=j;
            }
            return result;
        }
        if(k>n)
            throw std::invalid_argument("k must be less than or equal to n for distinct=true");
        else if(k<choose_parameters::threshold*n)
        {
            std::vector<NodeType> result;
            std::unordered_set<NodeType> v_set;
            while(v_set.size()<k)
            {
                std::uniform_int_distribution<NodeType> dist(0,n-1);
                auto j=dist(rng);
                v_set.insert(j);
            }
            for(auto i:v_set)
                result.push_back(i);
            return result;
        }
        else
        {
            std::vector<NodeType> result(n);
            for(int i=0;i<n;i++)
                result[i]=i;
            return choose(result,k,rng,distinct);
        }
    }

    DiscreteUniformWeightGenerator::DiscreteUniformWeightGenerator(size_t a, size_t b, std::uint64_t seed): rng(seed),std::uniform_int_distribution<size_t>(a,b)
    {
    }


    WeightType DiscreteUniformWeightGenerator::operator()() {
        return std::uniform_int_distribution<size_t>::operator()(rng);
    }

    WeightType NormalWeightGenerator::operator()() {
        return std::normal_distribution<WeightType>::operator()(rng);
    }

    NormalWeightGenerator::NormalWeightGenerator(WeightType mean, WeightType std, std::uint64_t seed) : rng(seed),std::normal_distribution<WeightType>(mean,std)
    {
    }

    UniformWeightGenerator::UniformWeightGenerator(WeightType a, WeightType b, std::uint64_t seed): rng(seed),std::uniform_real_distribution<WeightType>(a,b)
    {
    }

    WeightType UniformWeightGenerator::operator()()
    {
        return std::uniform_real_distribution<WeightType>::operator()(rng);
    }


    GnpGenerator::GnpGenerator(std::uint64_t n, double p, std::uint64_t seed): n(n),p(p),rng(seed)
    {

    }

    GraphType GnpGenerator::operator()()
    {
        GraphType G(n);
        std::binomial_distribution<NodeType> degree_distribution(n,p);
        for(NodeType u=0;u<n;u++)
        {
            auto d=degree_distribution(rng);
            G[u]= choose(n,d,rng);
        }
        return G;
    }

    WeightedGraphGenerator::WeightedGraphGenerator(std::shared_ptr<GraphGenerator> graph_generator,
                                                   std::shared_ptr<WeightGenerator> weight_generator) : graph_generator(std::move(graph_generator)),
                                                                                                        weight_generator(std::move(weight_generator))
    {
    }

    WeightedGraphType WeightedGraphGenerator::operator()()
    {
        auto G=graph_generator->operator()();
        WeightedGraphType W(G.size());
        for(size_t u=0;u<G.size();u++) for(auto v:G[u])
            W[u].emplace(v,weight_generator->operator()());
        return W;
    }

    GeneratorEnvironmentFactory::GeneratorEnvironmentFactory(std::shared_ptr<WeightedGraphGenerator> weighted_graph_generator, std::uint64_t seed) : weighted_graph_generator(std::move(weighted_graph_generator)),
                                                                                                                                                               rng(seed)
    {

    }

    std::shared_ptr<Environment> GeneratorEnvironmentFactory::NewEnvironment(const MPGMetaGame &metaGame)
    {
        auto G=weighted_graph_generator->operator()();
        std::uniform_int_distribution<NodeType> dist(0,G.size()-1);
        return std::make_shared<Environment>(G, dist(rng));
    }

    UniformGnpEnvironmentFactory::UniformGnpEnvironmentFactory(NodeType n, WeightType p, WeightType a, WeightType b, std::uint64_t seed): GeneratorEnvironmentFactory(
                    std::make_shared<WeightedGraphGenerator>(std::make_shared<SinklessGnpGenerator>(n,p,seed),
                    std::make_shared<UniformWeightGenerator>(a,b,seed)),
                    seed
                    )
    {

    }

    std::shared_ptr<Environment> ExampleEnvironmentFactory::NewEnvironment(const MPGMetaGame& metaGame) {
        auto G=WeightedGraphType::from_string(R"(1 2 5
2 3 -7
3 7 0
3 6 5
6 1 -3
1 4 4
4 5 -3
5 6 3
5 7 0
7 1 0
0 1 5)");
        return std::make_shared<Environment>(G, 0);
    }

    SinklessGnpGenerator::SinklessGnpGenerator(std::uint64_t n, double p, std::uint64_t seed) : n(n),p(p),rng(seed)
    {
    }

    GraphType SinklessGnpGenerator::operator()()
    {
        GraphType G(n);
        std::binomial_distribution<NodeType> degree_distribution(n,p);
        for(NodeType u=0;u<n;u++)
        {
            NodeType d=0;
            while(d==0)
                d=degree_distribution(rng);
            G[u]= choose(n,d,rng);
        }
        return G;
    }

    std::shared_ptr<const Game> ParserMetaFactory::CreateGame(const GameParameters &params) 
    {
        std::string game_generator=params.at("generator").string_value();
        if(game_generator=="example")
        {
            return std::make_shared<MPGMetaGame>(params, std::make_unique<ExampleEnvironmentFactory>());
        }
        else if(game_generator == "gnp")
        {
            std::string generator_params=params.at("generator_params").string_value();
            std::stringstream ss(generator_params);
            try {
                ss.exceptions(std::ios::failbit);
                std::uint64_t n;
                double p,a,b;
                ss>>n>>p >> a >> b;
                std::uint64_t seed=0;
                if(!ss.eof())
                    ss >> seed;
                return std::make_shared<MPGMetaGame>(params, std::make_unique<UniformGnpEnvironmentFactory>(n,p,a,b,seed));
            }
            catch (std::ios::failure& e)
            {
                throw std::invalid_argument("Invalid generator_params for gnp: "+generator_params);
            }

        }
        else if(game_generator == "file")
        {
            throw std::invalid_argument("file not implemented");
        }
        else if(game_generator == "folder")
        {
            throw std::invalid_argument("folder not implemented");
        }
        else if (game_generator == "specification")
        {
            throw std::invalid_argument("specification not implemented");
        }
    }
}