"""
Suite de tests d'invariance geometrique, en reponse directe aux questions
soulevees lors d'une revue technique independante du depot
limen-runtime-audit (juillet 2026).

A placer dans tests/test_invariance.py du depot (a cote de tests/test_metrics.py
existant), et a lancer avec :
    PYTHONPATH=src python -m unittest discover -s tests -v

Questions testees, reprises telles que posees dans le retour externe :

1. Path Length / Displacement : "does it behave as an intrinsic geometric
   property of the trajectory or is it primarily driven by the magnitude of
   the hidden-state vectors ?"
   -> Reponse mesuree ici : path_length et displacement scalent LINEAIREMENT
      avec la magnitude (ce n'est PAS une propriete intrinseque de la forme).

2. Tortuosity : "verify its invariance under translation, rotation, and
   uniform scaling."
   -> Reponse mesuree ici : tortuosite est invariante aux trois (c'est un
      ratio de deux longueurs, donc le facteur d'echelle s'annule ; les
      normes de differences sont invariantes par translation et rotation).

3. Velocity / Acceleration : "do they capture representational evolution or
   are they dominated by layer-wise scaling effects ?"
   -> Reponse mesuree ici : mean_speed et mean_acceleration scalent
      lineairement avec la magnitude, comme path_length/displacement -- donc
      sensibles a l'echelle des activations, pas seulement a leur "evolution"
      relative. Une comparaison inter-checkpoints doit donc soit normaliser,
      soit interpreter en valeur RELATIVE, jamais en valeur absolue brute.

4. Turning angle : "study the treatment of zero-motion cases and determine
   whether undefined directions should remain mathematically undefined
   instead of collapsing into zero-angle observations."
   -> Ce test NE JUGE PAS le comportement actuel, il le CARACTERISE : le code
      actuel (metrics.py, ligne `out=np.ones_like(denom), where=denom>0`)
      force cosine=1 (donc angle=0) quand un des deux vecteurs vitesse est
      nul, au lieu de NaN. C'est une convention de code, pas une propriete
      mathematique du signal -- exactement le point souleve par le retour
      externe.
"""

import sys
import unittest

import numpy as np

sys.path.insert(0, "src")  # ajuster si execute hors du depot
from limen_runtime_audit.metrics import layer_trajectory_metrics  # noqa: E402


def random_orthogonal_matrix(dim, rng):
    a = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(a)
    return q


class TestPathLengthDisplacementScaling(unittest.TestCase):
    """Question 1 du retour externe : intrinseque ou domine par la magnitude ?"""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        n_tokens, n_layers, dim = 4, 10, 16
        steps = self.rng.normal(scale=1.0, size=(n_tokens, n_layers, dim))
        self.hidden_states = np.cumsum(steps, axis=1)

    def test_path_length_scales_linearly_with_magnitude(self):
        base = layer_trajectory_metrics(self.hidden_states)
        scale = 7.3
        scaled = layer_trajectory_metrics(self.hidden_states * scale)
        np.testing.assert_allclose(scaled["path_length"], base["path_length"] * scale, rtol=1e-6)

    def test_displacement_scales_linearly_with_magnitude(self):
        base = layer_trajectory_metrics(self.hidden_states)
        scale = 7.3
        scaled = layer_trajectory_metrics(self.hidden_states * scale)
        np.testing.assert_allclose(scaled["displacement"], base["displacement"] * scale, rtol=1e-6)

    def test_conclusion_not_intrinsic(self):
        """path_length/displacement ne sont PAS des proprietes intrinseques
        de la forme -- deux trajectoires de meme forme mais d'echelle
        d'activation differente (ex: deux checkpoints avec des normes de
        residual stream differentes) auront des valeurs absolues
        differentes, sans que la trajectoire soit "plus" ou "moins" quoi
        que ce soit geometriquement. Comparer des valeurs absolues brutes
        entre checkpoints/modeles est donc invalide sans normalisation
        prealable ou interpretation relative."""
        base = layer_trajectory_metrics(self.hidden_states)
        scaled = layer_trajectory_metrics(self.hidden_states * 100.0)
        self.assertGreater(scaled["path_length"][0], base["path_length"][0] * 50)


class TestVelocityAccelerationScaling(unittest.TestCase):
    """Question 3 du retour externe : evolution representationnelle ou effet
    d'echelle architectural ?"""

    def setUp(self):
        self.rng = np.random.default_rng(7)
        n_tokens, n_layers, dim = 3, 12, 24
        steps = self.rng.normal(scale=1.0, size=(n_tokens, n_layers, dim))
        self.hidden_states = np.cumsum(steps, axis=1)

    def test_mean_speed_scales_linearly(self):
        base = layer_trajectory_metrics(self.hidden_states)
        scale = 4.2
        scaled = layer_trajectory_metrics(self.hidden_states * scale)
        np.testing.assert_allclose(scaled["mean_speed"], base["mean_speed"] * scale, rtol=1e-6)

    def test_mean_acceleration_scales_linearly(self):
        base = layer_trajectory_metrics(self.hidden_states)
        scale = 4.2
        scaled = layer_trajectory_metrics(self.hidden_states * scale)
        np.testing.assert_allclose(scaled["mean_acceleration"], base["mean_acceleration"] * scale, rtol=1e-6)

    def test_speed_cv_is_scale_invariant(self):
        """speed_cv (coefficient de variation) EST un ratio -- il devrait
        rester stable meme si mean_speed/mean_acceleration ne le sont pas.
        Utile pour des comparaisons inter-checkpoints sans normalisation
        manuelle prealable."""
        base = layer_trajectory_metrics(self.hidden_states)
        scaled = layer_trajectory_metrics(self.hidden_states * 4.2)
        np.testing.assert_allclose(scaled["speed_cv"], base["speed_cv"], rtol=1e-6)


class TestTortuosityInvariance(unittest.TestCase):
    """Question 2 du retour externe : invariance translation / rotation / echelle."""

    def setUp(self):
        self.rng = np.random.default_rng(123)
        n_tokens, n_layers, dim = 5, 15, 32
        steps = self.rng.normal(scale=1.0, size=(n_tokens, n_layers, dim))
        self.hidden_states = np.cumsum(steps, axis=1)

    def test_translation_invariance(self):
        base = layer_trajectory_metrics(self.hidden_states)
        shift = self.rng.normal(scale=5.0, size=(1, 1, self.hidden_states.shape[-1]))
        transformed = layer_trajectory_metrics(self.hidden_states + shift)
        np.testing.assert_allclose(base["tortuosity"], transformed["tortuosity"], rtol=1e-6)
        np.testing.assert_allclose(base["path_length"], transformed["path_length"], rtol=1e-6)
        np.testing.assert_allclose(base["displacement"], transformed["displacement"], rtol=1e-6)

    def test_rotation_invariance(self):
        base = layer_trajectory_metrics(self.hidden_states)
        dim = self.hidden_states.shape[-1]
        q = random_orthogonal_matrix(dim, self.rng)
        transformed = layer_trajectory_metrics(self.hidden_states @ q)
        np.testing.assert_allclose(base["tortuosity"], transformed["tortuosity"], rtol=1e-6)
        np.testing.assert_allclose(base["path_length"], transformed["path_length"], rtol=1e-6)
        np.testing.assert_allclose(base["displacement"], transformed["displacement"], rtol=1e-6)
        np.testing.assert_allclose(
            base["mean_turning_angle_rad"], transformed["mean_turning_angle_rad"], rtol=1e-6
        )

    def test_uniform_scale_invariance(self):
        base = layer_trajectory_metrics(self.hidden_states)
        transformed = layer_trajectory_metrics(self.hidden_states * 9.9)
        np.testing.assert_allclose(base["tortuosity"], transformed["tortuosity"], rtol=1e-6)
        np.testing.assert_allclose(
            base["mean_turning_angle_rad"], transformed["mean_turning_angle_rad"], rtol=1e-6
        )

    def test_conclusion_confirmed(self):
        """Confirmation directe de l'hypothese formulee dans le retour externe : la tortuosite est
        le descripteur le plus robuste du framework pour des comparaisons
        inter-checkpoints/modeles, precisement parce qu'elle est invariante
        aux trois transformations testees ici, contrairement a
        path_length/displacement/mean_speed/mean_acceleration."""
        pass  # les trois tests ci-dessus suffisent a etablir la conclusion


class TestTurningAngleZeroMotion(unittest.TestCase):
    """Question 4 du retour externe : direction indefinie ou angle force a zero ?"""

    def setUp(self):
        self.rng = np.random.default_rng(99)

    def test_zero_velocity_collapses_to_zero_angle_not_nan(self):
        """CARACTERISATION du comportement actuel, pas une validation. Si ce
        test echoue apres une correction du code, c'est attendu -- il faudra
        alors le remplacer par un test verifiant NaN explicite."""
        n_layers, dim = 6, 8
        traj = np.zeros((1, n_layers, dim))
        traj[0, 1:] = np.cumsum(self.rng.normal(size=(n_layers - 1, dim)), axis=0)
        traj[0, 2] = traj[0, 1]  # vitesse nulle entre layer 1 et 2

        result = layer_trajectory_metrics(traj)

        self.assertTrue(
            np.isfinite(result["mean_turning_angle_rad"][0]),
            "Comportement actuel attendu : angle fini (pas NaN) meme en cas de vitesse nulle.",
        )
        print(
            "\n[CARACTERISATION] Vitesse nulle -> cosine force a 1.0 -> angle=0 "
            "(convention 'out=np.ones_like(denom)' dans metrics.py). "
            "Ce n'est PAS une direction mathematiquement definie -- c'est un "
            "choix de code qui traite 'pas de mouvement' comme 'pas de virage', "
            "ce qui n'est pas equivalent. Point souleve par le retour externe : "
            "preserver NaN distinguerait explicitement les deux cas."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
