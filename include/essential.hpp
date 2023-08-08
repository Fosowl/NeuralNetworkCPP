
#ifndef ESSENCE_H
#define ESSENCE_H

#include <iostream>
#include <exception>
#include <vector>
#include <cmath>
#include <algorithm>

#define ROUND(X) X + (X & 1)

#define DEBUG_ENABLED true

#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"


#define DBG(X) if (DEBUG_ENABLED) std::cout << YELLOW << "[Debug] " << RESET << X << std::endl;
#define ERROR(X) if (DEBUG_ENABLED) std::cerr << RED << "[Error] " << RESET << X << std::endl;
#define WARNING(X) if (DEBUG_ENABLED) std::cout << YELLOW << "[Warning] " << RESET << X << std::endl;
#define PRINT(X) std::cout << CYAN << X << RESET << std::endl;
#define ABORT(err) if (ERR_ABORT) exit(err);

#endif