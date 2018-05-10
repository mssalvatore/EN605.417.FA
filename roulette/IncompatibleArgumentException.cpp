#include "IncompatibleArgumentException.h"
#include <string>
#include <stdio.h>

IncompatibleArgumentException::IncompatibleArgumentException(std::string msg): msg(msg) {}
IncompatibleArgumentException::~IncompatibleArgumentException() throw() {}

const char* IncompatibleArgumentException::what() const throw()
{
    return msg.c_str();
}
