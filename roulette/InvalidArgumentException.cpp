#include "InvalidArgumentException.h"
#include <string>
#include <stdio.h>

InvalidArgumentException::InvalidArgumentException(const char * argument, std::string type): argument(argument), type(type) {}
InvalidArgumentException::~InvalidArgumentException() throw() {}

const char* InvalidArgumentException::what() const throw()
{
    char *buffer = new char[256];
    snprintf(buffer, 255, "This provided argument is not a valid %s: %s", this->type.c_str(), this->argument);
    return buffer;
}
