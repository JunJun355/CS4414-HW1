#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include "alglibmisc.h"
#include <nlohmann/json.hpp>
#include <chrono>


using json = nlohmann::json;

void runBenchmark(alglib::kdtree& tree, alglib::real_1d_array& query, int k) {
    std::cout << "\n#### Starting Benchmark ####\n" << std::endl;

    const std::vector<double> epsilons = {0.0, 0.5, 1.0, 2.0, 5.0, 7.0, 8.0, 10.0, 20.0};
    const int num_trials = 3;

    alglib::kdtreequeryaknn(tree, query, k, 0.0);
    alglib::integer_1d_array exact_tags;
    alglib::kdtreequeryresultstags(tree, exact_tags);
    std::unordered_set<int> ground_truth_ids;
    for(alglib::ae_int_t i = 0; i < k; ++i) {
        ground_truth_ids.insert(exact_tags[i]);
    }

    // --- 2. Run Benchmark for each Epsilon ---
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::setw(10) << "Epsilon" << std::setw(20) << "Search Time (ms)" << std::setw(20) << "Accuracy" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (double eps : epsilons) {
        std::vector<double> trial_times;
        
        // Run trials to get median time
        for (int i = 0; i < num_trials; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            alglib::kdtreequeryaknn(tree, query, k, eps);
            auto end = std::chrono::high_resolution_clock::now();
            trial_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        std::sort(trial_times.begin(), trial_times.end());
        double median_time = trial_times[num_trials / 2];

        // Run once more to get results for accuracy calculation
        alglib::ae_int_t count = alglib::kdtreequeryaknn(tree, query, k, eps);
        alglib::integer_1d_array result_tags;
        alglib::kdtreequeryresultstags(tree, result_tags);
        
        int correct_neighbors = 0;
        for (alglib::ae_int_t i = 0; i < count; ++i) {
            if (ground_truth_ids.count(result_tags[i])) {
                correct_neighbors++;
            }
        }
        
        std::string accuracy_str = std::to_string(correct_neighbors) + "/" + std::to_string(k);
        std::cout << std::setw(10) << eps << std::setw(20) << median_time << std::setw(20) << accuracy_str << std::endl;
    }
    std::cout << "\n#### Benchmark Finished ####\n" << std::endl;
}

int main(int argc, char* argv[]) {
    auto program_start = std::chrono::high_resolution_clock::now();

    if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <query.json> <passages.json> <K> <eps>\n";
    return 1;
    }

    auto processing_start = std::chrono::high_resolution_clock::now();
    // Load and parse query JSON
    std::ifstream query_ifs(argv[1]);
    if (!query_ifs) {
        std::cerr << "Error opening query file: " << argv[1] << "\n";
        return 1;
    }
    json query_json;
    query_ifs >> query_json;
    if (!query_json.is_array() || query_json.size() < 1) {
        std::cerr << "Query JSON must be an array with at least 1 element\n";
        return 1;
    }

    // Load and parse passages JSON
    std::ifstream passages_ifs(argv[2]);
    if (!passages_ifs) {
        std::cerr << "Error opening passages file: " << argv[2] << "\n";
        return 1;
    }
    json passages_json;
    passages_ifs >> passages_json;
    if (!passages_json.is_array() || passages_json.size() < 1) {
        std::cerr << "Passages JSON must be an array with at least 1 element\n";
        return 1;
    }


    // Convert JSON array to a dict mapping id -> element
    std::unordered_map<int, json> dict;
    for (auto &elem : passages_json) {
        int id = elem["id"].get<int>();
        dict[id] = elem;
    }


    // Parse K and eps
    int k = std::stoi(argv[3]);
    double eps = std::stof(argv[4]);

    try{
        // Extract the query embedding
        auto query_obj   = query_json[0];
        size_t D         = query_obj["embedding"].size();
        alglib::real_1d_array query;
        query.setlength(D);
        for (size_t d = 0; d < D; ++d) {
            query[d] = query_obj["embedding"][d].get<double>();
        }
        /*
        TODO:
        1. Extract the passage embedding and store it in alglib::real_2d_array, store the idx of each embedding in alglib::integer_1d_array
        2. Build the KD-tree (alglib::kdtree) from the passages embeddings using alglib::buildkdtree
        3. Perform the k-NN search using alglib::knnsearch
        4. Query the results
            - Get the index of each found neighbour  using alglib::kdtreequeryresultstags
            - Get the distance between each found neighbour and the query embedding using alglib::kdtreequeryresultsdists
        */
        size_t N = passages_json.size();
        alglib::real_2d_array allPoints;
        allPoints.setlength(N, D);
        alglib::integer_1d_array tags;
        tags.setlength(N);

        // std::vector<std::pair<T, int>> allPoints;
        // allPoints.reserve(passages_json.size());
        // for (const auto& elem : passages_json) {
        //     T emb;
        //     if constexpr (std::is_same_v<T, float>) {
        //         emb = elem["embedding"].get<float>();
        //     } else {
        //         emb.resize(Embedding_T<T>::Dim());
        //         for (size_t i = 0; i < Embedding_T<T>::Dim(); ++i) {
        //             emb[i] = elem["embedding"][i].get<float>();
        //         }
        //     }
        //     int idx = elem["id"].get<int>();
        //     allPoints.emplace_back(emb, idx);
        // }

        for (size_t i = 0; i < N; ++i) {
            const auto& elem = passages_json[i];
            tags[i] = elem["id"].get<int>();
            for (size_t d = 0; d < D; ++d) {
                allPoints[i][d] = elem["embedding"][d].get<double>();
            }
        }

        auto processing_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> processing_duration = processing_end - processing_start;

        auto buildtree_start = std::chrono::high_resolution_clock::now();
        alglib::kdtree tree;
        alglib::kdtreebuildtagged(allPoints, tags, N, D, 0, 2, tree);
        auto buildtree_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> buildtree_duration = buildtree_end - buildtree_start;

        runBenchmark(tree, query, k);

        auto query_start = std::chrono::high_resolution_clock::now();
        alglib::ae_int_t count = alglib::kdtreequeryaknn(tree, query, k, eps);
        auto query_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> query_duration = query_end - query_start;

        alglib::real_1d_array dists;
        alglib::integer_1d_array result_tags;
        
        alglib::kdtreequeryresultsdistances(tree, dists);
        alglib::kdtreequeryresultstags(tree, result_tags);
        
        auto program_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> program_duration = program_end - program_start;


        std::cout << "query:\n";
        std::cout << "  text:    " << query_obj["text"] << "\n\n";
        
        std::cout << std::fixed << std::setprecision(6); // Format output for distances

        for (alglib::ae_int_t i = 0; i < count; ++i) {
            int   id   = result_tags[i];
            double dist = dists[i];
            auto& elem = dict[id];

            std::cout << "Neighbor " << (i + 1) << ":\n";
            std::cout << "  id:      " << id
                      << ", dist = " << dist << "\n";
            std::cout << "  text:    " << elem["text"].get<std::string>() << "\n\n";
        }

        std::cout << "#### Performance Metrics ####\n";
        std::cout << "Elapsed time: " << program_duration.count() << " ms\n";
        std::cout << "Processing time: " << processing_duration.count() << " ms\n";
        std::cout << "KD-tree build time: " << buildtree_duration.count() << " ms\n";
        std::cout << "K-NN query time: " << query_duration.count() << " ms\n";
    }
    catch(alglib::ap_error &e) {
        std::cerr << "ALGLIB error: " << e.msg << std::endl;
        return 1;
    }

    return 0;
}