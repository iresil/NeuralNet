#include "pch.h"
#include "path_provider.h"
#include <windows.h>
#include <cstdlib>


std::string PathProvider::safe_getenv(const char *name)
{
    char *buffer = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buffer, &len, name) == 0 && buffer != nullptr)
    {
        std::string value(buffer);
        free(buffer); // Free memory allocated by _dupenv_s
        return value;
    }
    return std::string();
}

bool PathProvider::is_visual_studio()
{
    // Environment Variables set at runtime
    std::string vs_env = safe_getenv("VSINSTALLDIR");
    bool is_env_var_present = vs_env.c_str() != nullptr;

    // Might be true for non-VS debuggers, like WinDbg
    bool is_debugger_present = IsDebuggerPresent();

    return is_env_var_present && is_debugger_present;
}

std::string PathProvider::get_full_path(std::string filename)
{
    std::filesystem::path folderPath;

    if (is_visual_studio())
    {
        folderPath = std::filesystem::path(__FILE__).parent_path().parent_path();
    }
    else
    {
        char path[MAX_PATH];
        DWORD length = GetModuleFileNameA(nullptr, path, MAX_PATH);
        if (length == 0)
        {
            return std::string();
        }

        std::filesystem::path exePath(path);
        folderPath = exePath.parent_path();
    }
    folderPath.append(filename);
    return folderPath.string();
}
