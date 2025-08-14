#pragma once
#include <string>

class PathProvider
{
    private:
        static std::string safe_getenv(const char *name);
        static bool is_visual_studio();

    public:
        static std::string get_full_path(std::string filename);
};
