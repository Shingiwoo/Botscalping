from executors.sim_exec import SimExecutor


def test_sim_exec_open_close():
    sim = SimExecutor({"ADAUSDT": 20.0})
    assert abs(sim.get_balance("ADAUSDT") - 20.0) < 1e-9
    sim.open_position("ADAUSDT", "LONG", 1.00, 10.0)
    assert sim.has_position("ADAUSDT")
    pnl = sim.close_position("ADAUSDT", 1.02)  # +0.02 * 10 = +0.2
    assert abs(pnl - 0.2) < 1e-9

