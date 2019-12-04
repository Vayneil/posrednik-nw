import numpy as np
import csv

with open('file.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)


def get_loop(bv_positions, ev_position):
    def inner(loop):
        if len(loop) > 3:
            can_be_closed = len(get_possible_next_nodes(loop, [ev_position])) == 1
            if can_be_closed:
                return loop

        not_visited = list(set(bv_positions) - set(loop))
        possible_next_nodes = get_possible_next_nodes(loop, not_visited)
        for next_node in possible_next_nodes:
            new_loop = inner(loop + [next_node])
            if new_loop:
                return new_loop

    return inner([ev_position])


def get_possible_next_nodes(loop, not_visited):
    last_node = loop[-1]
    nodes_in_row = [n for n in not_visited if n[0] == last_node[0]]
    nodes_in_column = [n for n in not_visited if n[1] == last_node[1]]
    if len(loop) < 2:
        return nodes_in_row + nodes_in_column
    else:
        prev_node = loop[-2]
        row_move = prev_node[0] == last_node[0]
        if row_move:
            return nodes_in_column
        return nodes_in_row


def same_line(arr, lowest_number):
    result = sum(map(lambda x: x == np.NINF or x == lowest_number, arr))
    if result > 1:
        return True
    else:
        return False


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


# Writing to file
file = open("wyniki.txt", "w")

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
file.write("Pierwsze rozwiązanie metodą wierzchołka północno-zachodniego:\n")
file.writelines(str(solution))
file.write("\n\n")

optimal = False
iteration = 0

while not optimal:
    # Optimality test using MODI method (Modified Distribution method)
    # Step 1: alpha and beta values
    iteration += 1
    str_iter = "###### ITERACJA: " + str(iteration) + " ######"
    file.write(str_iter)
    file.write("\n\n")
    solution_rows = solution.shape[0]
    solution_cols = solution.shape[1]
    alpha = np.empty(solution_rows)
    beta = np.empty(solution_cols)
    alpha[0] = 0
    base_profit = np.full(profit.shape, np.NINF)
    optimality = np.full(profit.shape, np.NINF)
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
            if base_profit[i][j] != np.NINF:
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

    file.write("ALFA:\n")
    file.writelines(str(alpha))
    file.write("\n\n")

    file.write("BETA:\n")
    file.writelines(str(beta))
    file.write("\n\n")

    optimal = True

    for i in range(0, solution_rows):
        for j in range(0, solution_cols):
            optimality[i][j] -= alpha[i] + beta[j]
            if optimality[i][j] > 0:
                optimal = False

    file.write("ZMIENNE KRYTERIALNE: \n")
    file.writelines(str(optimality))
    file.write("\n\n")

    bv_positions = []
    for i in range(0, solution_rows):
        for j in range(0, solution_cols):
            if optimality[i][j] == np.NINF:
                bv_positions.append((i, j))

    if optimal:
        file.write("ROZWIĄZANIE OPTYMALNE! \n")
        file.write("ZYSK: " + str(all_profit(solution, profit)))
        break

    least_optimal = np.unravel_index(optimality.argmax(), optimality.shape)

    cycle = get_loop(bv_positions, least_optimal)

    file.write("CYKL: \n")
    file.writelines(str(cycle))
    file.write("\n\n")

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
            if solution[cycle[i]] > 0:
                subtract_minimum.append(solution[cycle[i]])

    subtract_minimum = np.amin(subtract_minimum)

    for i in range(0, len(cycle)):
        if add_n_subtract[i]:
            solution[cycle[i]] += subtract_minimum
        else:
            solution[cycle[i]] -= subtract_minimum

    file.write("NOWY WYNIK:\n")
    file.writelines(str(solution))
    file.write("\n\n")

file.close()
