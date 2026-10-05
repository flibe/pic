# =============================================================================
#  Process Identification and Control - Sapienza, DIAG - Prof. Francesco Liberati
#
#  Service area control with MPC: the simulation loop.
#
#  A charging area draws power from the grid to recharge electric vehicles. An
#  energy storage system (ESS) is used to keep the power at the point of
#  connection (POC) small, which is what the capacity-based part of the bill is
#  paid on. Both the charging demand (ppev) and the renewable generation (pres)
#  are assumed known over the prediction window.
#
#  Sign convention (consumer convention): power consumed is positive, power
#  flowing out of an object is negative. So u > 0 charges the storage.
#
#      p^POC_k = ppev_k + u_k - pres_k
#      x_{k+1} = x_k + (T/3600) * u_k
#
#  This file runs the receding horizon loop; MPC_iteration.jl builds and solves
#  the optimization problem at each step. Same structure as the browser version:
#  https://flibe.github.io/pic/files/sim/service-area-lab.html
#
#  Run with:   julia main.jl
#  Needs:      JuMP, HiGHS, Plots   (see the Software setup page of the course)
# =============================================================================

using JuMP
using HiGHS                       # open source, installed in the course setup
# using Gurobi                    # free academic licence, usually faster
using Plots
using Printf

include("MPC_iteration.jl")

const OPTIMIZER = HiGHS.Optimizer
# const OPTIMIZER = Gurobi.Optimizer

# ----------------------------------------------------------------- parameters
par = (
    T      = 60.0,      # sampling time [s]
    N      = 20,        # prediction horizon [steps]
    alpha  = 1.0,       # weight on the power at the POC
    beta   = 0.02,      # weight on the stored energy reference
    beta_f = 0.02,      # terminal weight
    u_min  = -10.0,     # ESS power [kW], negative = discharging
    u_max  =  10.0,
    x_min  =   0.0,     # stored energy [kWh]
    x_max  = 100.0,
    x_ref  =  50.0,     # energy we like to keep in the storage [kWh]
    p_min  = -30.0,     # power at the POC [kW]
    p_max  =  30.0,
)

k_end = 24 * 60                   # 24 hours at one step per minute
x_1   = 30.0                      # initial energy in the storage [kWh]

# ------------------------------------------------- known profiles (forecasts)
# Charging demand of the vehicles. A quiet night, a morning wave, a long
# afternoon session: the kind of profile that makes the POC peak.
ppev = zeros(k_end + par.N)
ppev[8*60  : 9*60]  .= 12.0
ppev[9*60  : 10*60] .= 22.0
ppev[12*60 : 13*60] .=  8.0
ppev[17*60 : 19*60] .= 20.0

# Renewable generation on site, a bell shape centred at midday.
pres = zeros(k_end + par.N)
for k in 1:(k_end + par.N)
    hour = (k - 1) / 60
    pres[k] = max(0.0, 14.0 * sinpi((hour - 7.0) / 11.0))
end

# ------------------------------------------------------------------ logs
log_u = zeros(k_end)              # control applied [kW]
log_p = zeros(k_end)              # power at the POC [kW]
log_x = zeros(k_end + 1)          # stored energy [kWh]
log_t = zeros(k_end)              # solution time of each iteration [s]
log_x[1] = x_1

# --------------------------------------------------------------- MPC loop
for k in 1:k_end
    x_k = log_x[k]                                    # measure the state

    t = @elapsed u1, p1, _ = mpc_iteration(k, x_k, ppev, pres, par, OPTIMIZER)
    log_u[k], log_p[k], log_t[k] = u1, p1, t

    # apply the first control only and simulate the plant for one step
    log_x[k+1] = log_x[k] + par.T / 3600.0 * log_u[k]

    if k % 60 == 0
        @printf("%02d:00  x = %5.1f kWh   u = %6.2f kW   p = %6.2f kW\n",
                k ÷ 60, log_x[k+1], log_u[k], log_p[k])
    end
end

# ------------------------------------------------------------------ results
net = ppev[1:k_end] .- pres[1:k_end]      # what the POC would see without the ESS
@printf("\npeak |p| at the POC with MPC:        %6.2f kW\n", maximum(abs.(log_p)))
@printf("peak |p| at the POC without storage: %6.2f kW\n", maximum(abs.(net)))
@printf("mean solution time per iteration:    %6.1f ms\n", 1000 * sum(log_t) / k_end)

# ------------------------------------------------------------------ figures
hours = (0:k_end-1) ./ 60

p1 = plot(hours, log_p, label = "with MPC", lw = 2, ylabel = "POC power [kW]",
          legend = :topleft, color = RGB(0.18, 0.40, 0.56))
plot!(p1, hours, net, label = "without storage", lw = 1.5, ls = :dash, color = :grey40)
hline!(p1, [par.p_min, par.p_max], label = "limits", ls = :dash, color = :firebrick)

p2 = plot(hours, log_x[1:k_end], label = "stored energy", lw = 2,
          ylabel = "ESS energy [kWh]", color = RGB(0.23, 0.49, 0.27))
hline!(p2, [par.x_ref], label = "reference", ls = :dot, color = :seagreen)
hline!(p2, [par.x_min, par.x_max], label = "limits", ls = :dash, color = :firebrick)

p3 = plot(hours, log_u, label = "ESS power", lw = 1.5, ylabel = "u [kW]",
          seriestype = :steppost, color = RGB(0.66, 0.42, 0.07))
hline!(p3, [par.u_min, par.u_max], label = "limits", ls = :dash, color = :firebrick)

p4 = plot(hours, ppev[1:k_end], label = "charging demand", lw = 1.5, color = :grey30,
          xlabel = "time [h]", ylabel = "disturbances [kW]")
plot!(p4, hours, pres[1:k_end], label = "renewables", lw = 1.5, color = :darkorange)

fig = plot(p1, p2, p3, p4, layout = (4, 1), size = (900, 1000), xticks = 0:2:24)
savefig(fig, "service_area.png")
println("figure written to service_area.png")
