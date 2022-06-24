#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#include "gl_helper.h"

/*������һ������ CPUBitmap�Ľṹ�� */

/*��struct �� class ��������˿��Կ���http://blog.csdn.Net/tigernana/article/details/7606773 */

struct CPUBitmap

{
        unsigned char    *pixels;       /*���ص���ܸ���*/
        int     x, y;                              /*ͼ��ĳ���*/
        void    *dataBlock;                /*  */

       /*��voidָ�벻̫��Ϥ���ˣ����Բο� http://pcedu.pconline.com.cn/empolder/gj/c/0509/702366_all.html  */
        void (*bitmapExit)(void*);     /*����һ������ */


        /*��C����Ӱ�죬��struct���ɻ�ģ���ο��� http://leeing.org/2010/01/31/struct-vs-class-in-cpp/  */
        CPUBitmap( int width, int height, void *d = NULL )

        {
                pixels = new unsigned char[width * height * 4];   /*�����ܵ����ص������������µĿռ�*/
                x = width;                                                                      /*ͼ��Ŀ�*/
                y = height;                                                                     /*ͼ��ĸ�*/
               dataBlock = d;                                                                /* */                                           
        }

        /*��������*/
        ~CPUBitmap()

        {

              /*ɾ�����ص�*/

              delete [] pixels;       
         }

        /*ȡ���������ص�*/       

        unsigned char* get_ptr( void ) const   { return pixels; }

        /*ȡ��ͼƬ�ܴ�С*/

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

                      /*glutInit,�� GLUT (OpenGl �����һ�����߰��������ܶຯ��)���г�ʼ��,������������������� GLUTʹ��֮ǰ����һ�Ρ����ʽ�Ƚ�����,һ���ճ����glutInit(&argc, argv)�Ϳ�����*/

                      glutInit( &c, &dummy );        

                      /*������ʾ��ʽ,���� GLUT_RGBA ��ʾʹ�� RGBA ��ɫ,��֮��Ӧ�Ļ���GLUT_INDEX(��ʾʹ��������ɫ) ��GLUT_SINGLE ��ʾʹ�õ�����,����֮��Ӧ�Ļ��� GLUT_DOUBLE(ʹ��˫����)��*/    
                      glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

                      /*���Ҳ��,���ô��ڵĴ�С*/

                      glutInitWindowSize( x, y );

                      /*����ǰ�����õ���Ϣ�������ڡ�����������Ϊ���ڵı��⡣ע��:���ڱ�������,����������ʾ����Ļ�ϡ���Ҫ���� glutMainLoop ���ܿ������ڡ�*/

                      glutCreateWindow( "bitmap" );

                     /* http://tieba.baidu.com/p/296729172  ������ͨ��������������Key����*/   
                      glutKeyboardFunc(Key);

                      /* ����һ������,����Ҫ���л�ͼʱ,��������ͻᱻ���á�*/
                      glutDisplayFunc(Draw);

                      /*��ʾ����*/

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

                    /* �������������Esc���������˳�����*/


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

        /* ��ͼ */
        static void Draw( void )

        {
                 CPUBitmap*   bitmap = *(get_bitmap_ptr());

                 /*���ñ�����ɫ*/
                 glClearColor( 0.0, 0.0, 0.0, 1.0 );

                 /*�����GL_COLOR_BUFFER_BIT ��ʾ�����ɫ*/
                 glClear( GL_COLOR_BUFFER_BIT );

                 /* http://blog.csdn.net/ghost129/article/details/4409565 �������ص� */

                 glDrawPixels( bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );

                 /*��֤ǰ��� OpenGL ��������ִ��(�������������ڻ������еȴ�)�������ø� fflush(stdout)���ơ�*/
                 glFlush();
        }
};

#endif  // __CPU_BITMAP_H__