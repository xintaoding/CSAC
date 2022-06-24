/*看不懂头两行和最后一行是做什么事情的时候，参考 http://blog.csdn.NET/fulva/article/details/8208312  */

#ifndef __GL_HELPER_H__
#define __GL_HELPER_H__

/*
   在64位的Windows系统中, 我们需要防止 GLUT 自动链接glut32. 所以我们定义了 GLUT_NO_LIB_PRAGMA. 这意味着我们需要通过 pragmas手动添加opengl32.lib 和 glut64.lib回到原来的link. 这也可以通过 compilation/link 命令行实现, 但是，我们选择这个使编译在32位或者64为windows上都是一样的。（我自己也不知道自己翻译了什么，总之这段是说，如果用的系统是是64位的，就执行上面的一段的代码，手动加载两个lib；如果使用的是32位的WIN，则编译的时候编译的是下面那段代码，即包含了很多opengl的头文件。最后如果用的是linux的话，编译的是最下面的头文件）
*/
#ifdef _WIN64
#define GLUT_NO_LIB_PRAGMA
#pragma comment (lib, "opengl32.lib")  /* link with Microsoft OpenGL lib */
#pragma comment (lib, "glut64.lib")    /* link with Win64 GLUT lib */
#endif //_WIN64


#ifdef _WIN32
/* On Windows, include the local copy of glut.h and glext.h */
#include "GL/glut.h"
#include "GL/glext.h"

#define GET_PROC_ADDRESS( str ) wglGetProcAddress( str )

#else

/* On Linux, include the system's copy of glut.h, glext.h, and glx.h */
#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glx.h>

#define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte *)str )

#endif //_WIN32


#endif //__GL_HELPER_H__'

/*总结：这段代码是好意，让我们用他的代码的时候无论在什么环境下，都能运行，阿弥陀佛，善哉善哉！*/