//
// Created by ramizouari on 06/06/23.
//

#include "mpg_generator.h"

namespace open_spiel::mpg
{

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

    GeneratorMetaFactory::GeneratorMetaFactory(std::shared_ptr<WeightedGraphGenerator> weighted_graph_generator, std::uint64_t seed) : weighted_graph_generator(std::move(weighted_graph_generator)),
        rng(seed)
    {

    }

    std::shared_ptr<const Game> GeneratorMetaFactory::CreateGame(const GameParameters &params)
    {
        auto G=weighted_graph_generator->operator()();
        std::uniform_int_distribution<NodeType> dist(0,G.size()-1);
        return std::make_shared<MPGGame>(params,G,dist(rng));
    }

    UniformGnpMetaFactory::UniformGnpMetaFactory(NodeType n, WeightType p,WeightType a,WeightType b, std::uint64_t seed): GeneratorMetaFactory(
                    std::make_shared<WeightedGraphGenerator>(std::make_shared<GnpGenerator>(n,p,seed),
                    std::make_shared<UniformWeightGenerator>(a,b,seed)),
                    seed
                    )
    {

    }

    std::shared_ptr<const Game> ExampleFactory::CreateGame(const GameParameters &params) {
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
        return std::make_shared<MPGGame>(params,G,0);
    }
}