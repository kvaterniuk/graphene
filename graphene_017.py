# -*- coding: utf-8 -*-

# Цей код призначений для запуску в Google Colab.
# Перед виконанням, переконайтеся, що ви обрали "TPU" як тип апаратного прискорювача
# в меню "Runtime > Change runtime type".

# Імпортуємо необхідні бібліотеки.
# JAX для ефективних обчислень на TPU/GPU.
# NumPy для числових операцій.
# Matplotlib для візуалізації.
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Перевіряємо, чи доступний пристрій TPU.
try:
    print("Доступні пристрої JAX:", jax.devices())
    assert jax.device_count() > 0, "TPU не знайдено. Переконайтеся, що обрано правильний тип середовища."
except Exception as e:
    print(f"Помилка: {e}")
    print("TPU не знайдено або виникла помилка підключення.")

# ---- 1. Визначення моделі та констант ----

# Параметри моделі сильного зв'язку
t = -2.7  # Параметр перенесення (eV)
a_cc = 1.42  # Відстань між атомами вуглецю (Angstroms)
a = jnp.sqrt(3) * a_cc  # Постійна гратки

# Зміщений потенціал підґратки (моделює вертикальний ріст).
# Змініть це значення, щоб побачити, як змінюється зонна щілина.
delta = 0.5  # eV

# Визначаємо вектори до найближчих сусідів (1NN)
delta1 = jnp.array([a_cc * jnp.sqrt(3) / 2, a_cc / 2])
delta2 = jnp.array([a_cc * jnp.sqrt(3) / 2, -a_cc / 2])
delta3 = jnp.array([-a_cc * jnp.sqrt(3), 0])
neighbors_1nn = jnp.array([delta1, delta2, delta3])

# ---- 2. Гамільтоніан та обчислення енергій ----

# Функція для обчислення недіагонального елемента Гамільтоніана.
# Без використання декоратора jax.jit.
def h12_k_raw(k_point):
    """Обчислює елемент h12(k) Гамільтоніана."""
    return -t * jnp.sum(jnp.exp(1j * jnp.dot(k_point, neighbors_1nn.T)))

# Явно компілюємо функцію за допомогою jax.jit.
h12_k = jax.jit(h12_k_raw)

# Функція для обчислення енергетичних зон (власних значень).
# Без використання декоратора jax.jit.
def calculate_bands_raw(hamiltonian_func, k_points, delta_val):
    """Обчислює власні значення (енергії) для масиву k-точок."""
    # Застосовуємо jax.vmap явним викликом
    h_values = jax.vmap(hamiltonian_func)(k_points)
    # Енергії є власними значеннями модифікованого гамільтоніана:
    # E(k) = +/- sqrt(|h12(k)|^2 + delta^2)
    energies = jnp.array([-jnp.sqrt(jnp.abs(h_values)**2 + delta_val**2),
                          jnp.sqrt(jnp.abs(h_values)**2 + delta_val**2)]).T
    return energies

# Явно компілюємо функцію за допомогою jax.jit, вказуючи статичний аргумент.
calculate_bands = jax.jit(calculate_bands_raw, static_argnums=(0,))

# ---- 3. Шлях у k-просторі ----

# Визначаємо точки високої симетрії
gamma_point = jnp.array([0.0, 0.0])
K_point_fixed = (2 * jnp.pi / (3 * a)) * jnp.array([1, 1 / jnp.sqrt(3)])
M_point_fixed = (2 * jnp.pi / (3 * a)) * jnp.array([1, 0])

# Створюємо шлях у k-просторі для візуалізації (Gamma-K-M-Gamma)
num_points_path = 500
path_gk = jnp.linspace(gamma_point, K_point_fixed, num_points_path)
path_km = jnp.linspace(K_point_fixed, M_point_fixed, num_points_path)
path_mg = jnp.linspace(M_point_fixed, gamma_point, num_points_path)
k_path = jnp.concatenate((path_gk, path_km, path_mg))

# ---- 4. Обчислення та візуалізація ----

print("Розрахунок зонної структури для вертикально вирощеного графену...")
bands = calculate_bands(h12_k, k_path, delta)
bands_np = np.array(bands)
x_axis = np.linspace(0, 1, len(k_path))

plt.style.use('dark_background')
plt.figure(figsize=(10, 8))

# Побудова кривих енергетичних зон
plt.plot(x_axis, bands_np[:, 0], color='cyan', linewidth=2, label='Валентна зона')
plt.plot(x_axis, bands_np[:, 1], color='lime', linewidth=2, label='Зона провідності')

# Позначення зонної щілини
plt.fill_between(x_axis, bands_np[:, 0], bands_np[:, 1], color='grey', alpha=0.3)

# Додавання міток для точок високої симетрії на осі X.
total_len = len(k_path)
gk_len = len(path_gk)
km_len = len(path_km)

plt.xticks([0, gk_len / total_len, (gk_len + km_len) / total_len], ['Γ', 'K', 'M'])
plt.tick_params(axis='x', which='both', bottom=False, top=False)

# Додавання підписів та заголовка
plt.title('Зонна структура вертикально вирощеного графену', fontsize=18, color='white')
plt.xlabel('Шлях у просторі оберненої ґратки (k)', fontsize=14, color='white')
plt.ylabel('Енергія (еВ)', fontsize=14, color='white')
plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Енергія Фермі (E=0)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# Встановлюємо обмеження по осі Y
plt.ylim(-3.5, 3.5)

plt.show()