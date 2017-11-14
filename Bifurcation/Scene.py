from OpenGL.GL import *
from OpenGL.GLUT import *


def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    return shader

def draw():
    glClear(GL_COLOR_BUFFER_BIT)                    # Очищаем экран и заливаем серым цветом
    glEnableClientState(GL_VERTEX_ARRAY)            # Включаем использование массива вершин
    glEnableClientState(GL_COLOR_ARRAY)             # Включаем использование массива цветов
    # Указываем, где взять массив верши:
    # Первый параметр - сколько используется координат на одну вершину
    # Второй параметр - определяем тип данных для каждой координаты вершины
    # Третий парметр - определяет смещение между вершинами в массиве
    # Если вершины идут одна за другой, то смещение 0
    # Четвертый параметр - указатель на первую координату первой вершины в массиве
    # glVertexPointer(3, GL_FLOAT, 0, pointdata)
    # # Указываем, где взять массив цветов:
    # # Параметры аналогичны, но указывается массив цветов
    # glColorPointer(3, GL_FLOAT, 0, pointcolor)
    # # Рисуем данные массивов за один проход:
    # # Первый параметр - какой тип примитивов использовать (треугольники, точки, линии и др.)
    # # Второй параметр - начальный индекс в указанных массивах
    # # Третий параметр - количество рисуемых объектов (в нашем случае это 3 вершины - 9 координат)
    # glDrawArrays(GL_TRIANGLES, 0, 3)
    # glDisableClientState(GL_VERTEX_ARRAY)           # Отключаем использование массива вершин
    # glDisableClientState(GL_COLOR_ARRAY)            # Отключаем использование массива цветов
    glColor3f(0.8, 0.2, 0.1)
    glutSolidCone(0.5, 1, 50, 50)
    # glColor3f(0.5, 0.2, 0.4)
    # glutSolidCube(0.5)
    glutSwapBuffers()                               # Выводим все нарисованное в памяти на экран

# def mouse_func(button, state, x, y):
#     if(button == GLUT_LEFT_BUTTON):
#         print("okay")

# def mouseButton( button, mode, x, y ):
# 	"""Callback function (mouse button pressed or released).
#
# 	The current and old mouse positions are stored in
# 	a	global renderParam and a global list respectively"""
#
# 	global rP, oldMousePos
# 	if mode == GLUT_DOWN:
# 		rP.mouseButton = button
# 	else:
# 		rP.mouseButton = None
# 	oldMousePos[0], oldMousePos[1] = x, y
# 	glutPostRedisplay( )
#
# def mouseMotion( x, y ):
# 	"""Callback function (mouse moved while button is pressed).
#
# 	The current and old mouse positions are stored in
# 	a	global renderParam and a global list respectively.
# 	The global translation vector is updated according to
# 	the movement of the mouse pointer."""
#
# 	global rP, oldMousePos
# 	deltaX = x - oldMousePos[ 0 ]
# 	deltaY = y - oldMousePos[ 1 ]
# 	if rP.mouseButton == GLUT_LEFT_BUTTON:
# 		factor = 0.01
# 		rP.tVec[0] += deltaX * factor
# 		rP.tVec[1] -= deltaY * factor
# 		oldMousePos[0], oldMousePos[1] = x, y
# 	glutPostRedisplay( )

def specialkeys(key, x, y):
    # Сообщаем о необходимости использовать глобального массива pointcolor
    global pointcolor
    # Обработчики специальных клавиш
    if key == GLUT_KEY_UP:  # Клавиша вверх
        glRotatef(5, 1, 0, 0)       # Вращаем на 5 градусов по оси X
    if key == GLUT_KEY_DOWN:        # Клавиша вниз
        glRotatef(-5, 1, 0, 0)      # Вращаем на -5 градусов по оси X
    if key == GLUT_KEY_LEFT:        # Клавиша влево
        glRotatef(5, 0, 1, 0)       # Вращаем на 5 градусов по оси Y
    if key == GLUT_KEY_RIGHT:       # Клавиша вправо
        glRotatef(-5, 0, 1, 0)      # Вращаем на -5 градусов по оси Y
    if key == GLUT_KEY_END:         # Клавиша END
        # Заполняем массив pointcolor случайными числами в диапазоне 0-1
        pointcolor = [[random(), random(), random()], [random(), random(), random()], [random(), random(), random()]]

glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
glutInitWindowSize(500, 500)
glutInitWindowPosition(50, 50)
glutInit(sys.argv)
glutCreateWindow(b"Cone and Rect")
glutDialsFunc(draw)
glutIdleFunc(draw)
glutSpecialFunc(specialkeys)
glClearColor(0.2, 0.2, 0.2, 1)
glutMouseFunc(mouse_func)

#Создание вершинного шейдера
vertex  = create_shader(GL_VERTEX_SHADER, """
varying vec4 vertex_color;
            void main(){
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                vertex_color = gl_Color;
            }""")
fragment = create_shader(GL_FRAGMENT_SHADER, """
varying vec4 vertex_color;
            void main() {
                gl_FragColor = vertex_color;
}""")

program = glCreateProgram()
glAttachShader(program, vertex)
glAttachShader(program, fragment)
glLinkProgram(program)
glUseProgram(program)
pointdata = [[0, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0]]
# Определяем массив цветов (по одному цвету для каждой вершины)
pointcolor = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
# Запускаем основной цикл
glutMainLoop()

