"""M3 心跳的黄金测试：三态变心率 / 意图调度 / 主动性拍卖 / 反拗期。"""

import unittest

from intero.heartbeat import HEART_RATES, Heartbeat, HeartState, Intention


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestHeartRate(unittest.TestCase):
    def test_three_state_rates(self):
        clock = FakeClock()
        hb = Heartbeat(now=clock)
        hb.interact()
        self.assertEqual(hb.state, HeartState.ENGAGED)
        self.assertEqual(hb.rate, HEART_RATES[HeartState.ENGAGED])
        clock.advance(120)          # 2 分钟无交互 → 降值守
        hb.tick()
        self.assertEqual(hb.state, HeartState.WATCH)
        clock.advance(3600)         # 1 小时无交互 → 降休眠
        hb.tick()
        self.assertEqual(hb.state, HeartState.DREAM)
        hb.interact()               # 交互唤醒
        self.assertEqual(hb.state, HeartState.ENGAGED)

    def test_explicit_sleep_not_auto_exit(self):
        clock = FakeClock()
        hb = Heartbeat(now=clock)
        hb.sleep()
        clock.advance(10)
        hb.tick()
        self.assertEqual(hb.state, HeartState.DREAM)   # 休眠只能被交互唤醒


class TestIntentionScheduler(unittest.TestCase):
    def test_expired_intentions_gc(self):
        clock = FakeClock()
        hb = Heartbeat(now=clock)
        hb.add_intention(Intention(kind="remind", payload="喝水", urgency=0.9, ttl=10))
        clock.advance(20)
        hb.tick()
        self.assertEqual(len(hb.intentions), 0)

    def test_bid_shape(self):
        now = 1000.0
        it = Intention(kind="x", payload="", urgency=1.0, ttl=100, created_at=now)
        early = it.bid(now + 10)       # 前半程渐强
        peak = it.bid(now + 50)        # 中点峰值
        late = it.bid(now + 90)        # 后半程衰减
        self.assertLess(early, peak)
        self.assertLess(late, peak)


class TestAuction(unittest.TestCase):
    def test_silence_wins_when_bids_low(self):
        clock = FakeClock()
        hb = Heartbeat(now=clock, silence_floor=0.55)
        hb.add_intention(Intention(kind="weak", payload="", urgency=0.3, ttl=1000))
        r = hb.tick()
        self.assertEqual(r.winner, "silence")
        self.assertFalse(r.acted)
        self.assertEqual(len(hb.intentions), 1)   # 没行动的意图还在队列里

    def test_strong_intention_outbids_silence(self):
        clock = FakeClock()
        hb = Heartbeat(now=clock, silence_floor=0.55)
        # 预龄化到出价峰值（前半程渐强是设计：新意图不立刻吠）
        hb.add_intention(Intention(kind="urgent", payload="", urgency=0.95, ttl=1000,
                                   created_at=clock.t - 500))
        r = hb.tick()
        self.assertEqual(r.winner, "urgent")
        self.assertTrue(r.acted)
        self.assertEqual(len(hb.intentions), 0)   # 行动后出队

    def test_refractory_period_suppresses_action(self):
        clock = FakeClock()
        hb = Heartbeat(now=clock, silence_floor=0.5, refractory_ticks=2,
                       refractory_penalty=0.6)
        hb.add_intention(Intention(kind="a", payload="", urgency=0.9, ttl=1000,
                                   created_at=clock.t - 500))
        r1 = hb.tick()                            # 峰值出价 0.9 > 0.5 → 行动，进反拗期
        self.assertTrue(r1.acted)
        hb.add_intention(Intention(kind="b", payload="", urgency=0.9, ttl=1000,
                                   created_at=clock.t - 500))
        r2 = hb.tick()                            # 反拗期：0.9×0.4=0.36 < 0.5 → 沉默
        self.assertFalse(r2.acted)
        hb.tick()
        r4 = hb.tick()                            # 反拗期结束 → 行动
        self.assertTrue(r4.acted)
        self.assertEqual(r4.winner, "b")


if __name__ == "__main__":
    unittest.main()
