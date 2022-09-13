#ifndef DEME_HEADER_GENERATOR_H
#define DEME_HEADER_GENERATOR_H

#include <string>
#include <unordered_map>

class HeaderGenerator {
  public:
    std::string generateHeader(const std::filesystem::path&){

        // read source from file

        /*
            for pair in subst
                replace(source, pair.first, pair.second);
        */
    };

    template <typename T>
    void addSubstitution(std::string& key, T& value) {
        subst.emplace(key, std::to_string(value));
    }

  private:
    std::unordered_map<std::string, std::string> subst;
};

#endif