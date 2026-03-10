# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from jax.scipy.linalg import eigh as jax_eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Встановлення JAX для використання TPU
# Якщо ви запускаєте цей код на Google Colab, переконайтеся, що ви вибрали
# "TPU" як середовище виконання в "Change runtime type"

print(f"JAX backend: {jax.default_backend()}")

def create_tight_binding_model_3NN():
    """
    Створення моделі сильного зв'язку графену, що включає взаємодії до третього найближчого сусіда.
    Модель базується на 2x2 матриці Гамільтоніана в k-просторі, яка враховує
    як міжпідґраткові (1NN, 3NN), так і внутрішньопідґраткові (2NN) стрибки.
    Також включено інтеграли перекриття.
    """
    # Фізичні константи та параметри моделі
    a_cc = 1.42  # Å, відстань між атомами вуглецю
    a = a_cc * jnp.sqrt(3)  # Å, константа ґратки

    # Параметри стрибків (hopping parameters, γ)
    gamma0 = -2.70  # еВ
    gamma1 = -0.06  # еВ
    gamma2 = -0.095  # еВ

    # Інтеграли перекриття (overlap integrals, s)
    s0 = 0.086
    s1 = -0.015
    s2 = -0.004

    # Визначення векторів сусідів
    # Вектори до найближчих сусідів (1NN), з A-підґратки до B-підґратки
    delta1 = jnp.array([a_cc * jnp.sqrt(3)/2, a_cc * 1.5])
    delta2 = jnp.array([a_cc * jnp.sqrt(3)/2, a_cc * -1.5])
    delta3 = jnp.array([-a_cc * jnp.sqrt(3), 0.0])
    delta_1NN_vectors = jnp.stack([delta1, delta2, delta3])

    # Вектори до другого найближчого сусіда (2NN), всередині однієї підґратки (A->A, B->B)
    delta2_1 = jnp.array([0, a_cc*jnp.sqrt(3)])
    delta2_2 = jnp.array([a_cc*1.5, a_cc*jnp.sqrt(3)/2])
    delta2_3 = jnp.array([a_cc*1.5, -a_cc*jnp.sqrt(3)/2])
    delta2_4 = jnp.array([0, -a_cc*jnp.sqrt(3)])
    delta2_5 = jnp.array([-a_cc*1.5, -a_cc*jnp.sqrt(3)/2])
    delta2_6 = jnp.array([-a_cc*1.5, a_cc*jnp.sqrt(3)/2])
    delta_2NN_vectors = jnp.stack([delta2_1, delta2_2, delta2_3, delta2_4, delta2_5, delta2_6])

    # Вектори до третього найближчого сусіда (3NN), з A-підґратки до B-підґратки
    delta3_1 = jnp.array([-a_cc*jnp.sqrt(3)/2, a_cc*2.5])
    delta3_2 = jnp.array([-a_cc*jnp.sqrt(3)/2, -a_cc*2.5])
    delta3_3 = jnp.array([a_cc*jnp.sqrt(3), 0.0])
    delta_3NN_vectors = jnp.stack([delta3_1, delta3_2, delta3_3])

    def get_H_and_S(k):
        """
        Побудова матриць Гамільтоніана (H) та перекриття (S) в k-просторі.
        """
        f1 = jnp.dot(k, delta_1NN_vectors.T)
        f2 = jnp.dot(k, delta_2NN_vectors.T)
        f3 = jnp.dot(k, delta_3NN_vectors.T)

        H_AB = gamma0 * jnp.sum(jnp.exp(1j * f1)) + gamma2 * jnp.sum(jnp.exp(1j * f3))
        H_AA_BB = gamma1 * jnp.sum(jnp.exp(1j * f2))

        S_AB = s0 * jnp.sum(jnp.exp(1j * f1)) + s2 * jnp.sum(jnp.exp(1j * f3))
        S_AA_BB = 1.0 + s1 * jnp.sum(jnp.exp(1j * f2))

        H = jnp.array([[H_AA_BB, H_AB],
                      [jnp.conj(H_AB), H_AA_BB]])

        S = jnp.array([[S_AA_BB, S_AB],
                      [jnp.conj(S_AB), S_AA_BB]])

        return H, S

    return get_H_and_S

def get_k_path(n_points=100):
    """
    Генерує шлях у k-просторі вздовж ліній високої симетрії Γ-M-K-Γ.
    """
    a_cc = 1.42  # Å
    a = a_cc * jnp.sqrt(3)  # Å

    Gamma = jnp.array([0.0, 0.0])
    M = jnp.array([2*jnp.pi / (a*jnp.sqrt(3)), 0.0])
    K = jnp.array([2*jnp.pi / (a*jnp.sqrt(3)), 2*jnp.pi / (3*a)])

    path1_len = int(n_points * jnp.linalg.norm(M - Gamma) / (jnp.linalg.norm(M - Gamma) + jnp.linalg.norm(K - M) + jnp.linalg.norm(Gamma - K)))
    path2_len = int(n_points * jnp.linalg.norm(K - M) / (jnp.linalg.norm(M - Gamma) + jnp.linalg.norm(K - M) + jnp.linalg.norm(Gamma - K)))
    path3_len = n_points - path1_len - path2_len

    k_path1 = jnp.linspace(Gamma, M, path1_len)
    k_path2 = jnp.linspace(M, K, path2_len)
    k_path3 = jnp.linspace(K, Gamma, path3_len)

    k_path = jnp.concatenate([k_path1, k_path2, k_path3], axis=0)

    path_labels = [r'$\Gamma$', 'M', 'K', r'$\Gamma$']
    path_points = [0, path1_len-1, path1_len + path2_len - 1, n_points-1]

    return k_path, path_labels, path_points

def get_k_grid(n_points=100):
    """
    Генерує сітку k-точок для візуалізації у 3D.
    Ми створюємо квадратну сітку навколо K-точки, щоб візуалізувати
    конічні диракові точки.
    """
    a_cc = 1.42  # Å
    a = a_cc * jnp.sqrt(3)  # Å
    K = jnp.array([2*jnp.pi / (a*jnp.sqrt(3)), 2*jnp.pi / (3*a)])

    k_range = 0.5  # еВ, діапазон для сітки
    kx = jnp.linspace(K[0] - k_range, K[0] + k_range, n_points)
    ky = jnp.linspace(K[1] - k_range, K[1] + k_range, n_points)

    # Створення сітки
    kx_grid, ky_grid = jnp.meshgrid(kx, ky)
    k_grid = jnp.stack([kx_grid.flatten(), ky_grid.flatten()], axis=1)

    return kx_grid, ky_grid, k_grid

def main():
    # Отримання моделі сильного зв'язку
    get_H_and_S = create_tight_binding_model_3NN()

    # Оптимізація обчислення зонної структури за допомогою JAX
    @jax.jit
    @jax.vmap
    def compute_bands(k):
        H, S = get_H_and_S(k)

        # Перетворення узагальненої задачі на власні значення на стандартну
        # за допомогою Cholesky-розкладу матриці перекриття S.
        L = jnp.linalg.cholesky(S)
        L_inv = jnp.linalg.inv(L)
        H_prime = L_inv @ H @ L_inv.conj().T

        # Розв'язання стандартної задачі на власні значення
        eigenvalues = jax_eigh(H_prime, eigvals_only=True)

        return eigenvalues

    # --- Візуалізація 2D зонної структури вздовж шляху високої симетрії ---
    k_path, path_labels, path_points = get_k_path(n_points=500)
    print("Початок розрахунку 2D зонної структури...")
    energies_2d = compute_bands(k_path)
    print("Розрахунок 2D завершено.")

    plt.figure(figsize=(10, 6))
    plt.plot(energies_2d, 'b')  # Графік зонних структур

    for point in path_points:
        plt.axvline(x=point, color='gray', linestyle='--')

    plt.xticks(path_points, path_labels)
    plt.ylabel('Енергія (еВ)')
    plt.title('Електронна зонна структура графену (TB, 3NN)')
    plt.grid(True)
    plt.show()

    # --- Візуалізація 3D зонної структури в k-просторі ---
    n_grid_points = 50
    kx_grid, ky_grid, k_grid = get_k_grid(n_points=n_grid_points)
    print("Початок розрахунку 3D зонної структури...")
    energies_3d = compute_bands(k_grid)
    print("Розрахунок 3D завершено.")

    # Перетворення одновимірного масиву енергій на двовимірну сітку
    energy_band1 = energies_3d[:, 0].reshape(n_grid_points, n_grid_points)
    energy_band2 = energies_3d[:, 1].reshape(n_grid_points, n_grid_points)

    # Побудова 3D-графіку
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Побудова поверхонь для двох енергетичних зон (валентної та провідності)
    ax.plot_surface(kx_grid, ky_grid, energy_band1, cmap='viridis')
    ax.plot_surface(kx_grid, ky_grid, energy_band2, cmap='viridis')

    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.set_zlabel('Енергія (еВ)')
    ax.set_title('3D-візуалізація зонної структури графену')

    plt.show()

if __name__ == "__main__":
    main()