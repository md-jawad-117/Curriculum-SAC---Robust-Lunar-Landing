"""
Custom Visualization for Curriculum SAC — LunarLanderContinuous-v3
==================================================================
Custom visuals:
  - Starry night sky with twinkling
  - Lunar-style ground (craters, rocks, textured surface)
  - Custom lander with metallic body, animated flames
  - HUD overlay (reward, step, stage, thrust bars, velocity, legs)
  - Trajectory trail
  - Optional video recording

Usage:
    python render_custom.py --model models/sac_best.pt
    python render_custom.py --model models/sac_best.pt --stage 4 --record
"""

import argparse
import os
import math
import numpy as np
import gymnasium as gym
import pygame
from sac import SACAgent

# ─────────────────────── Stage configs ─────────────────────── #

STAGES = [
    {"name": "Stage 1 — Calm",          "enable_wind": False, "wind_power": 0.0,  "turbulence_power": 0.0},
    {"name": "Stage 2 — Light Wind",    "enable_wind": True,  "wind_power": 8.0,  "turbulence_power": 0.5},
    {"name": "Stage 3 — Moderate Wind", "enable_wind": True,  "wind_power": 15.0, "turbulence_power": 1.0},
    {"name": "Stage 4 — Heavy Wind",    "enable_wind": True,  "wind_power": 20.0, "turbulence_power": 2.0},
]


# ─────────────────────── Background Generators ─────────────────────── #

def generate_starfield(width, height, n_stars=120, seed=42):
    """Generate star positions with size, brightness, twinkle params."""
    rng = np.random.RandomState(seed)
    stars = []
    for _ in range(n_stars):
        x = rng.randint(0, width)
        y = rng.randint(0, int(height * 0.75))
        size = rng.choice([1, 1, 1, 2, 2, 3])
        brightness = rng.randint(140, 255)
        twinkle_speed = rng.uniform(0.02, 0.08)
        twinkle_offset = rng.uniform(0, 2 * math.pi)
        stars.append((x, y, size, brightness, twinkle_speed, twinkle_offset))
    return stars


def draw_starry_sky(surface, stars, frame_count):
    """Dark gradient sky with twinkling stars."""
    w, h = surface.get_size()
    for y in range(h):
        t = y / h
        r = int(5 + 10 * t)
        g = int(5 + 15 * t)
        b = int(25 + 20 * (1 - t))
        pygame.draw.line(surface, (r, g, b), (0, y), (w, y))

    for (sx, sy, size, brightness, speed, offset) in stars:
        twinkle = math.sin(frame_count * speed + offset)
        b = max(60, min(255, int(brightness + twinkle * 50)))
        color = (b, b, min(255, b + 10))
        if size <= 1:
            surface.set_at((sx, sy), color)
        else:
            pygame.draw.circle(surface, color, (sx, sy), size)
            if size >= 3:
                glow = pygame.Surface((size * 6, size * 6), pygame.SRCALPHA)
                pygame.draw.circle(glow, (*color, 30), (size * 3, size * 3), size * 3)
                surface.blit(glow, (sx - size * 3, sy - size * 3))


def generate_terrain_and_craters(width, ground_y, height, seed=42):
    """Generate terrain points and craters."""
    rng = np.random.RandomState(seed)
    terrain = []
    x = 0
    while x < width + 10:
        bump = rng.uniform(-2, 2)
        big = math.sin(x * 0.015) * 12 + math.sin(x * 0.007) * 8
        terrain.append((x, ground_y + bump + big))
        x += rng.randint(3, 12)

    craters = []
    for _ in range(4):
        cx = rng.randint(50, width - 50)
        cy = rng.randint(ground_y + 5, height - 10)
        radius = rng.randint(5, 12)
        depth = rng.randint(2, 5)
        craters.append((cx, cy, radius, depth))

    return terrain, craters


def draw_lunar_ground(surface, terrain, craters, ground_y):
    """Textured lunar surface with craters and rocks."""
    w, h = surface.get_size()
    base = (60, 55, 50)
    dark = (40, 38, 35)
    light = (80, 75, 68)

    polygon = terrain + [(w, h), (0, h)]
    pygame.draw.polygon(surface, base, polygon)

    for i in range(len(terrain) - 1):
        x1, y1 = terrain[i]
        x2, y2 = terrain[i + 1]
        pygame.draw.line(surface, light, (x1, y1), (x2, y2), 2)
        pygame.draw.line(surface, dark, (x1, y1 + 3), (x2, y2 + 3), 1)

    for (cx, cy, radius, depth) in craters:
        pygame.draw.ellipse(surface, light, (cx - radius, cy - radius // 2, radius * 2, radius), 1)
        ir = int(radius * 0.7)
        pygame.draw.ellipse(surface, dark, (cx - ir, cy - ir // 2 + depth, ir * 2, ir), 0)
        tr = max(2, radius // 4)
        pygame.draw.ellipse(surface, (50, 48, 44), (cx - tr, cy + depth, tr * 2, tr), 0)

    rng = np.random.RandomState(99)
    for _ in range(40):
        rx, ry = rng.randint(0, w), rng.randint(ground_y + 10, h - 5)
        rs = rng.randint(1, 4)
        rc = rng.randint(35, 75)
        pygame.draw.circle(surface, (rc, rc - 5, rc - 10), (rx, ry), rs)


def draw_landing_zone(surface, ground_y):
    """Landing pad with flag markers."""
    w = surface.get_size()[0]
    cx = w // 2
    pw = 80
    pygame.draw.rect(surface, (90, 85, 80), (cx - pw // 2, ground_y - 3, pw, 6))
    pygame.draw.rect(surface, (120, 115, 110), (cx - pw // 2, ground_y - 3, pw, 2))

    for fx in [cx - pw // 2 - 5, cx + pw // 2 + 5]:
        pygame.draw.line(surface, (180, 180, 180), (fx, ground_y - 3), (fx, ground_y - 35), 2)
        pts = [(fx, ground_y - 35), (fx + 15, ground_y - 30), (fx, ground_y - 25)]
        pygame.draw.polygon(surface, (220, 50, 50), pts)


def draw_lander(surface, x, y, angle, main_thrust, side_thrust, leg_l, leg_r):
    """Custom metallic lander with animated flames."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rot(px, py):
        return (int(x + px * cos_a + py * sin_a), int(y + (px * sin_a - py * cos_a)))

    bw, bh = 16, 20
    bh = -bh  # flip lander right-side up

    # Body
    body = [rot(-bw, -bh * 0.3), rot(-bw * 0.7, -bh), rot(bw * 0.7, -bh),
            rot(bw, -bh * 0.3), rot(bw * 0.8, bh * 0.5), rot(-bw * 0.8, bh * 0.5)]
    pygame.draw.polygon(surface, (190, 195, 210), body)
    pygame.draw.polygon(surface, (140, 145, 160), body, 2)

    # Blue accent stripe
    stripe = [rot(-bw * 0.9, -bh * 0.1), rot(bw * 0.9, -bh * 0.1),
              rot(bw * 0.85, bh * 0.1), rot(-bw * 0.85, bh * 0.1)]
    pygame.draw.polygon(surface, (70, 130, 200), stripe)

    # Window
    wx, wy = rot(0, -bh * 0.55)
    pygame.draw.circle(surface, (150, 220, 255), (wx, wy), 5)
    pygame.draw.circle(surface, (200, 240, 255), (wx - 1, wy - 1), 2)

    # Nozzle
    nozzle = [rot(-6, bh * 0.5), rot(6, bh * 0.5), rot(8, bh * 0.8), rot(-8, bh * 0.8)]
    pygame.draw.polygon(surface, (100, 100, 110), nozzle)

    # Legs
    for sign, leg_contact in [(-1, leg_l), (1, leg_r)]:
        ext = 14 if leg_contact else 10
        p1 = rot(sign * bw * 0.7, bh * 0.3)
        p2 = rot(sign * bw * 1.4, bh * 0.3 - ext)
        pygame.draw.line(surface, (160, 160, 170), p1, p2, 2)
        f1 = rot(sign * bw * 1.6, bh * 0.3 - ext)
        f2 = rot(sign * bw * 1.2, bh * 0.3 - ext)
        pygame.draw.line(surface, (160, 160, 170), f1, f2, 3)

    # Main flame
    if main_thrust > 0.05:
        fl = int(15 + main_thrust * 35)
        fw = int(4 + main_thrust * 8)
        # Add jitter for animation
        jx = np.random.randint(-2, 3)
        jl = fl + np.random.randint(-3, 4)

        outer = [rot(-fw, bh * 0.8), rot(fw, bh * 0.8), rot(jx, bh * 0.8 - jl)]

        pygame.draw.polygon(surface, (255, 120, 30), outer)

        il = int(jl * 0.6)
        iw = max(1, fw // 2)
        inner = [rot(-iw, bh * 0.8), rot(iw, bh * 0.8), rot(jx, bh * 0.8 - il)]
        pygame.draw.polygon(surface, (255, 255, 150), inner)

        # Sparks
        for _ in range(3):
            spx = np.random.randint(-fw, fw)
            spy = np.random.randint(int(fl * 0.3), fl)
            sp = rot(spx, bh * 0.8 - spy)

            pygame.draw.circle(surface, (255, 200, 50), sp, np.random.randint(1, 3))

    # Side flames
    if abs(side_thrust) > 0.05:
        sl = int(8 + abs(side_thrust) * 15)
        sign = 1 if side_thrust > 0 else -1

        sp1 = rot(sign * (bw + 2), -bh * 0.2)
        sp2 = rot(sign * (bw + sl), -bh * 0.2 - 3)
        sp3 = rot(sign * (bw + sl), -bh * 0.2 + 3)
        pygame.draw.polygon(surface, (200, 220, 255), [sp1, sp2, sp3])


# ─────────────────────── Main Visualization ─────────────────────── #

def run_custom_visualization(agent, stage, n_episodes=5, record=False, save_dir="results/videos"):
    env = gym.make(
        "LunarLanderContinuous-v3",
        render_mode="rgb_array",
        enable_wind=stage["enable_wind"],
        wind_power=stage["wind_power"],
        turbulence_power=stage["turbulence_power"],
    )

    WINDOW_W, WINDOW_H = 800, 600
    HUD_HEIGHT = 100
    TOTAL_H = WINDOW_H + HUD_HEIGHT
    GROUND_Y = int(WINDOW_H * 0.82)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, TOTAL_H))
    pygame.display.set_caption(f"Curriculum SAC — {stage['name']}")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("Consolas", 22, bold=True)
    font_med = pygame.font.SysFont("Consolas", 18)
    font_small = pygame.font.SysFont("Consolas", 14)

    BG_HUD = (15, 15, 25)
    WHITE = (255, 255, 255)
    GREEN = (0, 220, 100)
    RED = (220, 60, 60)
    YELLOW = (255, 220, 50)
    CYAN = (0, 200, 255)
    ORANGE = (255, 160, 40)

    stars = generate_starfield(WINDOW_W, WINDOW_H)
    terrain, craters = generate_terrain_and_craters(WINDOW_W, GROUND_Y, WINDOW_H)

    bg_surface = pygame.Surface((WINDOW_W, WINDOW_H))
    frames = []
    frame_count = 0

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        step = 0
        trajectory = []

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    env.close()
                    return

            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1
            frame_count += 1

            trajectory.append((state[0], state[1]))

            # Lander state
            lx, ly = state[0], state[1]
            lander_angle = state[4]
            leg_l, leg_r = state[6], state[7]
            screen_x = int((lx + 1) / 2 * WINDOW_W)
            screen_y = int((1 - (ly + 0.38) / 1.8) * WINDOW_H)

            state = next_state

            # ── Draw scene ──
            draw_starry_sky(bg_surface, stars, frame_count)
            draw_lunar_ground(bg_surface, terrain, craters, GROUND_Y)
            draw_landing_zone(bg_surface, GROUND_Y)

            # Trail
            if len(trajectory) > 1:
                trail_surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
                for i in range(1, len(trajectory)):
                    x1 = int((trajectory[i-1][0] + 1) / 2 * WINDOW_W)
                    x2 = int((trajectory[i][0] + 1) / 2 * WINDOW_W)
                    y1 = int((1 - (trajectory[i-1][1] + 0.4) / 1.8) * WINDOW_H)
                    y2 = int((1 - (trajectory[i][1] + 0.4) / 1.8) * WINDOW_H)
                    
                    a = max(40, int(200 * i / len(trajectory)))
                    pygame.draw.line(trail_surf, (100, 180, 255, a), (x1, y1), (x2, y2), 2)
                bg_surface.blit(trail_surf, (0, 0))

            # Lander
            main_t = max(0, action[0])
            side_t = action[1]
            draw_lander(bg_surface, screen_x, screen_y, lander_angle,
                       main_t, side_t, leg_l, leg_r)

            # ── Compose screen ──
            screen.fill(BG_HUD)
            screen.blit(bg_surface, (0, 0))
            pygame.draw.line(screen, (50, 50, 70), (0, WINDOW_H), (WINDOW_W, WINDOW_H), 2)

            # ── HUD ──
            hy = WINDOW_H + 5

            screen.blit(font_large.render(stage["name"], True, CYAN), (15, hy))
            screen.blit(font_med.render(f"Episode: {ep}/{n_episodes}  |  Step: {step}", True, WHITE), (15, hy + 28))
            r_color = GREEN if ep_reward > 0 else RED
            screen.blit(font_med.render(f"Reward: {ep_reward:.1f}", True, r_color), (15, hy + 52))
            screen.blit(font_small.render(f"Wind: {stage['wind_power']:.0f}  Turb: {stage['turbulence_power']:.1f}", True, YELLOW), (15, hy + 76))

            bx, bw, bh = 450, 200, 16

            # Main engine bar
            screen.blit(font_small.render("Main Engine:", True, WHITE), (bx, hy + 10))
            pygame.draw.rect(screen, (50, 50, 60), (bx + 120, hy + 10, bw, bh))
            fw = int(main_t * bw)
            bc = ORANGE if main_t > 0.5 else GREEN
            pygame.draw.rect(screen, bc, (bx + 120, hy + 10, fw, bh))

            # Side engine bar
            screen.blit(font_small.render("Side Engine:", True, WHITE), (bx, hy + 35))
            pygame.draw.rect(screen, (50, 50, 60), (bx + 120, hy + 35, bw, bh))
            mx = bx + 120 + bw // 2
            sw = int(abs(side_t) * bw / 2)
            if side_t > 0:
                pygame.draw.rect(screen, CYAN, (mx, hy + 35, sw, bh))
            else:
                pygame.draw.rect(screen, CYAN, (mx - sw, hy + 35, sw, bh))
            pygame.draw.line(screen, WHITE, (mx, hy + 33), (mx, hy + 53), 1)

            # Velocity
            vx, vy = next_state[2], next_state[3]
            screen.blit(font_small.render(f"Vx: {vx:+.2f}  Vy: {vy:+.2f}", True, WHITE), (bx, hy + 60))

            # Legs
            screen.blit(font_small.render("Legs: ", True, WHITE), (bx, hy + 78))
            pygame.draw.circle(screen, GREEN if leg_l else RED, (bx + 60, hy + 86), 6)
            pygame.draw.circle(screen, GREEN if leg_r else RED, (bx + 80, hy + 86), 6)
            screen.blit(font_small.render("L", True, WHITE), (bx + 55, hy + 72))
            screen.blit(font_small.render("R", True, WHITE), (bx + 75, hy + 72))

            # Status
            if terminated and ep_reward > 200:
                screen.blit(font_large.render("LANDED!", True, GREEN), (WINDOW_W - 150, hy + 10))
            elif terminated and ep_reward < -100:
                screen.blit(font_large.render("CRASHED!", True, RED), (WINDOW_W - 150, hy + 10))

            pygame.display.flip()
            clock.tick(60)

            if record:
                frames.append(np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2)))

        pygame.time.wait(1000)
        print(f"  Episode {ep}: reward = {ep_reward:.1f}, steps = {step}")

    pygame.quit()
    env.close()

    if record and frames:
        os.makedirs(save_dir, exist_ok=True)
        try:
            import imageio
            vpath = f"{save_dir}/{stage['name'].replace(' ', '_').replace('—','')}.mp4"
            imageio.mimsave(vpath, frames, fps=30)
            print(f"[✓] Video saved → {vpath}")
        except ImportError:
            print("Install imageio for video recording: pip install imageio[ffmpeg]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sac_curriculum_best.pt")
    parser.add_argument("--stage", type=int, default=1, help="Stage 1-4")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    env = gym.make("LunarLanderContinuous-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()

    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(args.model)
    print(f"Loaded: {args.model}")

    stage = STAGES[args.stage - 1]
    print(f"Running: {stage['name']}\n")

    run_custom_visualization(agent, stage, args.episodes, record=args.record)


if __name__ == "__main__":
    main()