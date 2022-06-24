#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#include "gl_helper.h"

/*创建了一个叫做 CPUBitmap的结构体 */

/*对struct 和 class 有困惑的人可以看：http://blog.csdn.Net/tigernana/article/details/7606773 */

struct CPUBitmap

{
        unsigned char    *pixels;       /*像素点的总个数*/
        int     x, y;                              /*图像的长宽*/
        void    *dataBlock;                /*  */

       /*对void指针不太熟悉的人，可以参看 http://pcedu.pconline.com.cn/empolder/gj/c/0509/702366_all.html  */
        void (*bitmapExit)(void*);     /*这是一个函数 */


        /*受C语言影响，对struct有疑惑的，请参看： http://leeing.org/2010/01/31/struct-vs-class-in-cpp/  */
        CPUBitmap( int width, int height, void *d = NULL )

        {
                pixels = new unsigned char[width * height * 4];   /*计算总的像素点数，并分配新的空间*/
                x = width;                                                                      /*图像的宽*/
                y = height;                                                                     /*图像的高*/
               dataBlock = d;                                                                /* */                                           
        }

        /*析构函数*/
        ~CPUBitmap()

        {

              /*删除像素点*/

              delete [] pixels;       
         }

        /*取得所有像素点*/       

        unsigned char* get_ptr( void ) const   { return pixels; }

        /*取得图片总大小*/

        long image_size( void ) const { return x * y * 4; }
 

       

        void display_and_exit( void(*e)(void*) = NULL )

        {
                      CPUBitmap**   bitmap = get_bitmap_ptr();
                      *bitmap = this;
                      bitmapExit = e;


                      // a bug in the Windows GLUT implementation prevents us from
                      // passing zero arguments to glutInit()
                      int c=1;
                      char* dummy = "";

                      /*glutInit,对 GLUT (OpenGl 里面的一个工具包，包含很多函数)进行初始化,这个函数必须在其它的 GLUT使用之前调用一次。其格式比较死板,一般照抄这句glutInit(&argc, argv)就可以了*/

                      glutInit( &c, &dummy );        

                      /*设置显示方式,其中 GLUT_RGBA 表示使用 RGBA 颜色,与之对应的还有GLUT_INDEX(表示使用索引颜色) ；GLUT_SINGLE 表示使用单缓冲,。与之对应的还有 GLUT_DOUBLE(使用双缓冲)。*/    
                      glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

                      /*这个也简单,设置窗口的大小*/

                      glutInitWindowSize( x, y );

                      /*根据前面设置的信息创建窗口。参数将被作为窗口的标题。注意:窗口被创建后,并不立即显示到屏幕上。需要调用 glutMainLoop 才能看到窗口。*/

                      glutCreateWindow( "bitmap" );

                     /* http://tieba.baidu.com/p/296729172  当有普通按键被按，调用Key函数*/   
                      glutKeyboardFunc(Key);

                      /* 设置一个函数,当需要进行画图时,这个函数就会被调用。*/
                      glutDisplayFunc(Draw);

                      /*显示窗口*/

                      glutMainLoop();

          }

         // static method used for glut callbacks
         static CPUBitmap** get_bitmap_ptr( void )

        {
                 static CPUBitmap   *gBitmap;
                 return &gBitmap;
         }

         // static method used for glut callbacks
        static void Key(unsigned char key, int x, int y)

        {

                    /* 如果按键按的是Esc按键，则退出程序。*/


                        switch (key)

                        {
                                      case 27:
                                      CPUBitmap*   bitmap = *(get_bitmap_ptr());
                                      if (bitmap->dataBlock != NULL && bitmap->bitmapExit != NULL)
                                                                    bitmap->bitmapExit( bitmap->dataBlock );
                                      exit(0);
                          }
           }

       // static method used for glut callbacks

        /* 画图 */
        static void Draw( void )

        {
                 CPUBitmap*   bitmap = *(get_bitmap_ptr());

                 /*设置背景颜色*/
                 glClearColor( 0.0, 0.0, 0.0, 1.0 );

                 /*清除。GL_COLOR_BUFFER_BIT 表示清除颜色*/
                 glClear( GL_COLOR_BUFFER_BIT );

                 /* http://blog.csdn.net/ghost129/article/details/4409565 绘制像素点 */

                 glDrawPixels( bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );

                 /*保证前面的 OpenGL 命令立即执行(而不是让它们在缓冲区中等待)。其作用跟 fflush(stdout)类似。*/
                 glFlush();
        }
};

#endif  // __CPU_BITMAP_H__