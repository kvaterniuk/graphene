# -*- coding: utf-8 -*-

# Цей код призначений для запуску в Google Colab.
# Переконайтеся, що ви вибрали "TPU" як тип апаратного прискорювача
# в меню "Runtime > Change runtime type".

# Імпортуємо необхідні бібліотеки.
# JAX є основною бібліотекою для роботи з TPU.
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
except:
    print("TPU не знайдено або виникла помилка підключення.")

# Визначаємо константи для моделі.
# a_cc: відстань між атомами вуглецю (в ангстремах).
a_cc = 1.42  # Angstroms
a = jnp.sqrt(3) * a_cc  # Постійна гратки
t1 = -2.7  # Параметр перенесення для 1NN (eV)
t2 = -0.2  # Параметр перенесення для 2NN (eV)
t3 = -0.05 # Параметр перенесення для 3NN (eV)

# Визначаємо вектори гратки сусідів.
# 1NN (найближчі сусіди):
delta1 = jnp.array([a_cc * jnp.sqrt(3) / 2, a_cc / 2])
delta2 = jnp.array([a_cc * jnp.sqrt(3) / 2, -a_cc / 2])
delta3 = jnp.array([-a_cc * jnp.sqrt(3), 0])
neighbors_1nn = jnp.array([delta1, delta2, delta3])

# 2NN (другі найближчі сусіди):
# Ці вектори з'єднують атоми однієї підгратки.
rho1 = jnp.array([0, a_cc * jnp.sqrt(3)])
rho2 = jnp.array([a_cc * 3 / 2, a_cc * jnp.sqrt(3) / 2])
rho3 = jnp.array([a_cc * 3 / 2, -a_cc * jnp.sqrt(3) / 2])
rho4 = jnp.array([0, -a_cc * jnp.sqrt(3)])
rho5 = jnp.array([-a_cc * 3 / 2, -a_cc * jnp.sqrt(3) / 2])
rho6 = jnp.array([-a_cc * 3 / 2, a_cc * jnp.sqrt(3) / 2])
neighbors_2nn = jnp.array([rho1, rho2, rho3, rho4, rho5, rho6])

# 3NN (треті найближчі сусіди):
# Ці вектори з'єднують атоми різних підграток.
eta1 = jnp.array([a_cc * jnp.sqrt(3), 0])
eta2 = jnp.array([-a_cc * jnp.sqrt(3) / 2, a_cc * 3 / 2])
eta3 = jnp.array([-a_cc * jnp.sqrt(3) / 2, -a_cc * 3 / 2])
eta4 = jnp.array([-a_cc * jnp.sqrt(3), 0])
eta5 = jnp.array([a_cc * jnp.sqrt(3) / 2, -a_cc * 3 / 2])
eta6 = jnp.array([a_cc * jnp.sqrt(3) / 2, a_cc * 3 / 2])
neighbors_3nn = jnp.array([eta1, eta2, eta3, eta4, eta5, eta6])


# Визначаємо точки високої симетрії в k-просторі.
gamma_point = jnp.array([0.0, 0.0])
K_point = jnp.array([4 * jnp.pi / (3 * jnp.sqrt(3) * a_cc), 0.0])
M_point = jnp.array([2 * jnp.pi / (3 * jnp.sqrt(3) * a_cc), 2 * jnp.pi / (3 * a_cc)])
# K_point and M_point definition
# K-point = (2pi/sqrt(3)a, 2pi/3a)
K_point_fixed = (2 * jnp.pi / (3 * a)) * jnp.array([1, 1 / jnp.sqrt(3)])
M_point_fixed = (2 * jnp.pi / (3 * a)) * jnp.array([1, 0])


# Створюємо шлях у k-просторі для візуалізації.
num_points_path = 500
path_gm = jnp.linspace(gamma_point, M_point_fixed, num_points_path)
path_mk = jnp.linspace(M_point_fixed, K_point_fixed, num_points_path)
path_kg = jnp.linspace(K_point_fixed, gamma_point, num_points_path)
k_path = jnp.concatenate((path_gm, path_mk, path_kg))


# Функція для Гамільтоніана моделі 1NN.
def hamiltonian_1nn(k):
    """Обчислює матрицю Гамільтоніана для моделі 1NN."""
    f_k = jnp.sum(jnp.exp(1j * jnp.dot(k, neighbors_1nn.T)))
    H = jnp.array([[0, t1 * f_k],
                   [t1 * jnp.conj(f_k), 0]])
    return H

# Функція для Гамільтоніана моделі 3NN.
def hamiltonian_3nn(k):
    """Обчислює матрицю Гамільтоніана для моделі 3NN, включаючи 2NN і 3NN."""
    # Терм від 1NN (діагональний елемент)
    f1_k = jnp.sum(jnp.exp(1j * jnp.dot(k, neighbors_1nn.T)))

    # Терм від 2NN (діагональні елементи)
    f2_k = jnp.sum(jnp.exp(1j * jnp.dot(k, neighbors_2nn.T)))

    # Терм від 3NN (діагональні елементи)
    f3_k = jnp.sum(jnp.exp(1j * jnp.dot(k, neighbors_3nn.T)))

    # Складання матриці Гамільтоніана для моделі 3NN.
    # Діагональні елементи: t2 терм
    # Недіагональні елементи: t1 + t3 терми
    H_ab = t1 * f1_k + t3 * f3_k
    H = jnp.array([[t2 * f2_k, H_ab],
                   [jnp.conj(H_ab), t2 * f2_k]])
    return H

# Функція для обчислення енергетичних зон.
def _calculate_bands(hamiltonian_func, k_points):
    """
    Обчислює енергетичні зони для масиву k-точок, використовуючи заданий Гамільтоніан.
    hamiltonian_func: Функція Гамільтоніана (наприклад, hamiltonian_1nn або hamiltonian_3nn).
    k_points: Масив k-точок.
    """
    # JAX.vmap дозволяє ефективно обробляти масиви.
    eigenvalues = jax.vmap(lambda k: jnp.linalg.eigvalsh(hamiltonian_func(k)))(k_points)
    return eigenvalues

# Обходимо проблему TypeError, викликаючи jax.jit безпосередньо.
calculate_bands = jax.jit(_calculate_bands, static_argnums=0)

print("Розрахунок зонної структури для моделі 1NN...")
bands_1nn = calculate_bands(hamiltonian_1nn, k_path)

print("Розрахунок зонної структури для моделі 3NN...")
bands_3nn = calculate_bands(hamiltonian_3nn, k_path)

# Перетворюємо результати на NumPy для візуалізації.
bands_1nn_np = np.array(bands_1nn)
bands_3nn_np = np.array(bands_3nn)
x_axis = np.linspace(0, 1, len(k_path))

# Візуалізуємо результати.
plt.style.use('dark_background')
plt.figure(figsize=(10, 8))

# Побудова кривих для обох моделей.
plt.plot(x_axis, bands_1nn_np[:, 0], color='cyan', linestyle='-', linewidth=2, label='Модель 1NN (Валентна зона)')
plt.plot(x_axis, bands_1nn_np[:, 1], color='cyan', linestyle='-', linewidth=2, label='Модель 1NN (Зона провідності)')
plt.plot(x_axis, bands_3nn_np[:, 0], color='lime', linestyle='--', linewidth=2, label='Модель 3NN (Валентна зона)')
plt.plot(x_axis, bands_3nn_np[:, 1], color='lime', linestyle='--', linewidth=2, label='Модель 3NN (Зона провідності)')

# Додаємо мітки для точок високої симетрії на осі X.
total_len = len(k_path)
gm_len = len(path_gm)
mk_len = len(path_mk)
kg_len = len(path_kg)

plt.xticks([0, gm_len / total_len, (gm_len + mk_len) / total_len], ['Γ', 'M', 'K'])
plt.tick_params(axis='x', which='both', bottom=False, top=False)

plt.title('Порівняння зонних структур графену', fontsize=18, color='white')
plt.xlabel('Шлях у просторі оберненої ґратки (k)', fontsize=14, color='white')
plt.ylabel('Енергія (eV)', fontsize=14, color='white')
plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Енергія Фермі (E=0)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=12)
plt.show()