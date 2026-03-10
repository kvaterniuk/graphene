import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# --- Параметри моделі ---
# t: Інтеграл перестрибування (hopping parameter) в еВ.
#    Це енергія, пов'язана з переходом електрона між сусідніми атомами.
t = 2.7

# a: Відстань між атомами вуглецю (C-C) в ангстремах.
a = 1.42

# --- Геометрія ґратки ---
# Вектори, що з'єднують атом підґратки A з трьома його найближчими сусідами (B).
d_vectors = a * jnp.array([
    [1.0, 0.0],
    [-0.5, jnp.sqrt(3)/2],
    [-0.5, -jnp.sqrt(3)/2]
])

# --- Зона Бріллюена ---
# Координати точок високої симетрії в оберненому просторі.
Gamma = jnp.array([0, 0])
K = jnp.array([2 * jnp.pi / (3 * a), 2 * jnp.pi / (3 * jnp.sqrt(3) * a)])
M = jnp.array([2 * jnp.pi / (3 * a), 0])

# Кількість точок на кожному відрізку шляху
N_points = 100

# Створення шляху k-точками: Γ -> M -> K -> Γ
k_path_gm = jnp.linspace(Gamma, M, N_points, endpoint=False)
k_path_mk = jnp.linspace(M, K, N_points, endpoint=False)
k_path_kg = jnp.linspace(K, Gamma, N_points, endpoint=False)
k_path = jnp.vstack([k_path_gm, k_path_mk, k_path_kg])

# Функція для побудови Гамільтоніану H(k) для однієї k-точки
@jax.jit
def get_hamiltonian(k):
    """
    Обчислює матрицю Гамільтоніану 2x2 для графену в заданій k-точці.
    """
    # H_ab = -t * Σ exp(i * k . d_j)
    off_diagonal = -t * jnp.sum(jnp.exp(1j * jnp.dot(d_vectors, k)))
    
    # Гамільтоніан є Ермітовою матрицею
    H = jnp.array([
        [0, off_diagonal],
        [jnp.conj(off_diagonal), 0]
    ])
    return H

# Функція для знаходження власних значень (енергій)
@jax.jit
def get_energies(k):
    """
    Обчислює власні значення (енергії) для Гамільтоніану H(k).
    """
    H = get_hamiltonian(k)
    # jnp.linalg.eigh повертає власні значення (реальні) для Ермітових матриць
    eigenvalues, _ = jnp.linalg.eigh(H)
    return eigenvalues

# --- Обчислення ---
# Використовуємо jax.vmap для векторизації обчислень по всьому шляху k-точок.
energies = jax.vmap(get_energies)(k_path)

# --- Візуалізація ---
print("Розрахунок завершено. Готуємо графік зонної структури...")

# Створення осі x для графіка
x_axis = jnp.arange(energies.shape[0])
plt.figure(figsize=(8, 6))

# Побудова двох енергетичних зон (валентної та зони провідності)
plt.plot(x_axis, energies[:, 0], color='b') # Нижня зона (валентна)
plt.plot(x_axis, energies[:, 1], color='r') # Верхня зона (провідності)

# Позначення точок симетрії
plt.xticks(
    [0, N_points, 2 * N_points, 3 * N_points],
    ['Γ', 'M', 'K', 'Γ']
)
plt.axvline(x=0, color='k', linestyle='--')
plt.axvline(x=N_points, color='k', linestyle='--')
plt.axvline(x=2 * N_points, color='k', linestyle='--')
plt.axvline(x=3 * N_points, color='k', linestyle='--')


# Налаштування графіка
plt.title('Зонна структура графену (модель сильного зв\'язку)')
plt.xlabel('Хвильовий вектор (шлях у зоні Бріллюена)')
plt.ylabel('Енергія (еВ)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, 3 * N_points)
plt.axhline(0, color='k', linestyle='-', linewidth=0.5) # Нульовий рівень енергії (Фермі)

# Відображення графіка
plt.show()