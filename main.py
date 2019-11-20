import numpy as np


def nw_corner(cost, demand, supply):
    solution = np.zeros_like(cost)
    num_of_rows = solution.shape[0]
    num_of_cols = solution.shape[1]
    row = 0
    col = 0
    while row < num_of_rows and col < num_of_cols:
        if supply[row] > demand[col]:
            solution[row][col] = demand[col]
            supply[row] -= demand[col]
            demand[col] = 0
            col += 1
        else:
            solution[row][col] = supply[row]
            demand[col] -= supply[row]
            supply[row] = 0
            row += 1

    return solution


def is_odd(arr):
    if any(i == 1 for i in arr):
        return True
    else:
        return False


def all_profit(solution, profit):
    result = 0
    for i in range(0, solution.shape[0]):
        for j in range(0, solution.shape[1]):
            result += solution[i][j] * profit[i][j]
    return result


# TODO: Dynamic arrays linked to GUI
# Initial data
cost = np.array([[8, 14, 17], [12, 9, 19]])
demand = np.array([10, 28, 27])
supply = np.array([20, 30])
buy = np.array([10, 12])
sell = np.array([30, 25, 30])

# Shapes of cost array
cost_rows = cost.shape[0]
cost_cols = cost.shape[1]

# Check if problem is balanced
# If it's not, add fake supplier and client with zero profit and transportation cost
demand_sum = np.sum(demand)
supply_sum = np.sum(supply)
balance = demand_sum - supply_sum
if balance:
    cost = np.hstack((cost, np.zeros((cost_rows, 1))))
    cost = np.vstack((cost, np.zeros((1, cost.shape[1]))))
    supply = np.append(supply, demand_sum)
    demand = np.append(demand, supply_sum)

# Table of profits
profit = np.zeros_like(cost)
for i in range(0, cost_rows):
    for j in range(0, cost_cols):
        profit[i][j] = sell[j] - cost[i][j] - buy[i]

# IBFS (Initial basic feasible solution) - North-West corner method
solution = nw_corner(cost, demand, supply)
print(solution)

optimal = False

while not optimal:
    # Optimality test using MODI method (Modified Distribution method)
    # Step 1: alpha and beta values
    solution_rows = solution.shape[0]
    solution_cols = solution.shape[1]
    alpha = np.empty(solution_rows)
    beta = np.empty(solution_cols)
    alpha[0] = 0
    base_profit = np.full(profit.shape, np.inf)
    optimality = np.full(profit.shape, np.inf)
    for i in range(0, solution_rows):
        for j in range(0, solution_cols):
            if solution[i][j] != 0:
                base_profit[i][j] = profit[i][j]
            else:
                optimality[i][j] = profit[i][j]

    a = []
    b = []
    num_of_equations = solution_rows + solution_cols - 1

    for i in range(0, solution_rows):
        for j in range(0, solution_cols):
            if base_profit[i][j] != np.inf:
                b.append(base_profit[i][j])
                c = np.zeros(num_of_equations)
                if i != 0:
                    c[i - 1] = 1
                c[solution_rows - 1 + j] = 1
                a.append(c)

    a = np.asarray(a)
    b = np.asarray(b)
    x = np.linalg.solve(a, b)

    for i in range(0, x.shape[0]):
        if i < alpha.shape[0] - 1:
            alpha[i + 1] = x[i]
        else:
            beta[i - alpha.shape[0] + 1] = x[i]

    print(alpha)
    print(beta)

    optimal = True

    for i in range(0, solution_rows):
        for j in range(0, solution_cols):
            optimality[i][j] -= alpha[i] + beta[j]
            if optimality[i][j] < 0:
                optimal = False

    if optimal:
        print(all_profit(solution, profit))
        break

    print(optimality)
    least_optimal = np.unravel_index(optimality.argmin(), optimality.shape)

    print(least_optimal)

    cycle = [least_optimal]
    even_rows = np.zeros(optimality.shape[0])
    even_cols = np.zeros(optimality.shape[1])

    even_rows[least_optimal[0]] += 1
    even_cols[least_optimal[1]] += 1

    recent_row = least_optimal[0]
    recent_col = least_optimal[1]

    print("eeeeeee")

    while is_odd(even_rows) or is_odd(even_cols):
        if is_odd(even_rows):
            # next_node_row = np.where(even_rows == 1)[0][0]
            next_node_row = recent_row
            col_candidates = np.where(even_cols < 2)[0]
            for c in col_candidates:
                if optimality[next_node_row][c] == np.inf and (next_node_row, c) != least_optimal:
                    next_node_col = c
                    recent_col = c
                    even_rows[next_node_row] += 1
                    even_cols[next_node_col] += 1
                    cycle.append((next_node_row, next_node_col))
                    print(cycle)
                    break
        else:
            next_node_col = recent_col
            row_candidates = np.where(even_rows < 2)[0]
            for c in row_candidates:
                if optimality[c][next_node_col] == np.inf and (c, next_node_col) != least_optimal:
                    next_node_row = c
                    recent_row = c
                    even_rows[next_node_row] += 1
                    even_cols[next_node_col] += 1
                    cycle.append((next_node_row, next_node_col))
                    print(cycle)
                    break
    print(cycle)

    least_optimal_index = cycle.index(least_optimal)
    add_n_subtract = []
    for i in range(len(cycle)):
        if (least_optimal_index + i) % 2 == 0:
            add_n_subtract.append(True)
        else:
            add_n_subtract.append(False)

    subtract_minimum = []
    for i in range(len(add_n_subtract)):
        if not add_n_subtract[i]:
            subtract_minimum.append(solution[cycle[i]])

    subtract_minimum = np.amin(subtract_minimum)

    for i in range(0, len(cycle)):
        if add_n_subtract[i]:
            solution[cycle[i]] += subtract_minimum
        else:
            solution[cycle[i]] -= subtract_minimum

    print(solution)















