//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl

__kernel void square(__global * buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] * buffer[id];
}
