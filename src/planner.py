# from scipy.interpolate import CubicSpline
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from scipy.integrate import cumulative_trapezoid as cumtrapz
# import time

# points = np.array([
#     [0, 0],
#     [40, 10],
#     [50, 60],
#     [0, 90],
#     [-40, 70],
#     [-60, 40],
#     [-70, -20]
# ])

# N = len(points)

# # Параметризация по длине хорды
# t = np.zeros(N)
# for i in range(1, N):
#     dist = np.linalg.norm(points[i] - points[i-1])
#     t[i] = t[i-1] + dist

# # Параметрические кубические сплайны
# cs_x = CubicSpline(t, points[:, 0], bc_type='natural')
# cs_y = CubicSpline(t, points[:, 1], bc_type='natural')
# cs_dx = cs_x.derivative()  # производная для скорости
# cs_dy = cs_y.derivative()

# t_fine = np.linspace(t[0], t[-1], 200)
# x_sp = cs_x(t_fine)
# y_sp = cs_y(t_fine)

# dx_du = cs_dx(t_fine)
# dy_du = cs_dy(t_fine)
# ds_du = np.sqrt(dx_du**2 + dy_du**2)
# L_u = cumtrapz(ds_du, t_fine, initial=0)
# total_length = L_u[-1]

# # Шаг 2: Обратная функция u(L)
# L_to_u = CubicSpline(L_u, t_fine)

# # Шаг 3: Задаем закон движения (постоянная скорость V)
# V = 10.0
# t_end = total_length / V
# t_points = np.linspace(0, t_end, 500)
# L_points = V * t_points
# u_t = L_to_u(np.clip(L_points, 0, total_length))

# # Итог: траектория во времени
# x_t = cs_x(u_t)
# y_t = cs_y(u_t)

# # print(u_t)
# print(t_end)
# print(total_length)
# t_end = 0
# print(f'current_x: {cs_x(L_to_u(np.clip(t_end*V, 0, total_length)))}')
# print(f'current_y: {cs_y(L_to_u(np.clip(t_end*V, 0, total_length)))}')
# dx = cs_dx(L_to_u(np.clip(t_end*V, 0, total_length)))
# dy = cs_dy(L_to_u(np.clip(t_end*V, 0, total_length)))
# norm = np.linalg.norm([dx, dy])
# print(f'current_dx: {dx/norm}')
# print(f'current_dy: {dy/norm}')
# feedforward = [dx*V/norm, dy*V/norm]

# # Создание фигуры для анимации
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.plot(points[:, 0], points[:, 1], 'ro', markersize=8, label='Опорные точки')
# ax.plot(x_sp, y_sp, 'g-', linewidth=2, label='Кубический сплайн')
# ax.set_xlim(-80, 60)
# ax.set_ylim(-30, 100)
# ax.set_aspect('equal')
# ax.grid(True, alpha=0.3)
# ax.legend()
# ax.set_xlabel('Ось Ox')
# ax.set_ylabel('Ось Oy')

# # Точка и вектор скорости
# moving_point, = ax.plot([], [], 'bo', markersize=12, label='Движущаяся точка')
# velocity_vector = ax.quiver([], [], [], [], angles='xy', scale_units='xy', 
#                               scale=0.3, color='red', width=0.008, label='Вектор скорости')

# def animate(frame):
#     t_param = t[0] + (t[-1] - t[0]) * (frame / 199)
    
#     x = cs_x(t_param)
#     y = cs_y(t_param)
    
#     # Вычисляем скорость (производную) в этой точке
#     vx = cs_dx(t_param)
#     vy = cs_dy(t_param)
    
#     moving_point.set_data([x], [y])
#     velocity_vector.set_offsets([x, y])
#     velocity_vector.set_UVC(vx, vy)
    
#     return moving_point, velocity_vector

# anim = FuncAnimation(fig, animate, init_func=lambda: (moving_point, velocity_vector), 
#                       frames=200, interval=20, blit=True)

# plt.show()

import mujoco
import mujoco.viewer
import numpy as np
import time

xml = """
<mujoco>
  <visual>
    <rgba haze="0 0 0 0"/>
  </visual>

  <worldbody>
    <light name="sun" directional="true" diffuse="0.9 0.9 0.9" pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.05" rgba="0.7 0.7 0.7 1"/>
    
    <!-- Маркер как mocap тело (управляется из кода) -->
    <body name="red_marker" mocap="true">
      <geom type="sphere" size="0.01" rgba="1 0 0 0.95"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        t = time.time() - start
        x = np.sin(t) * 1.5
        y = np.cos(t * 0.7) * 1.5
        z = 0.3 + np.sin(t * 2) * 0.2
        
        # Обновляем позицию mocap тела
        data.mocap_pos[0] = [x, y, z]
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.02)