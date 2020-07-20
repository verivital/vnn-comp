import torch
import copy

from plnn.proxlp_solver.utils import bdot

class ByLayerDecomposition:

    def __init__(self, init_method):
        initializers = {'KW': self.get_initial_kw_solution,
                        'naive': self.get_initial_naive_solution}
        assert init_method in initializers, "Unknown initialization type"
        self.initial_dual_solution = initializers[init_method]

    @staticmethod
    def get_optim_primal(weights, additional_coeffs,
                         lower_bounds, upper_bounds,
                         dual_vars):
        lambdas, rhos = dual_vars.lambdas, dual_vars.rhos

        zas = []
        zahats = []
        zbs = []
        zbhats = []

        # Optimize over all the A constraints
        for lay_idx, (lay, lb, ub) in enumerate(zip(weights,
                                                    lower_bounds, upper_bounds)):
            is_first_layer = (lay_idx == 0)
            if is_first_layer:
                lin_km1 = torch.zeros_like(lb)
            else:
                lin_km1 = lambdas[lay_idx-1]
            is_last_layer = (lay_idx == len(weights)-1)
            if is_last_layer:
                lin_hat_k = 0
            else:
                lin_hat_k = -rhos[lay_idx]
            if (lay_idx+1) in additional_coeffs:
                lin_hat_k += additional_coeffs[(lay_idx+1)]
            za_km1, zahat_k = opt_A_domain(lay, lin_km1, lin_hat_k,
                                           lb, ub, not is_first_layer)
            if not is_first_layer:
                zas.append(za_km1)
            else:
                z0 = za_km1
            zahats.append(zahat_k)

        # Optimize over all the B constraints:
        for lay_idx in range(len(weights[:-1])):
            lb_k = lower_bounds[1+lay_idx]
            ub_k = upper_bounds[1+lay_idx]
            lin_k = -lambdas[lay_idx]
            lin_hat_k = rhos[lay_idx]

            zbhat_k, zb_k = opt_B_domain(lin_hat_k, lin_k,
                                         lb_k, ub_k)
            zbhats.append(zbhat_k)
            zbs.append(zb_k)

        primal = PrimalVarSet(zas, zahats, zbs, zbhats, z0)

        return primal

    @staticmethod
    def get_optim_primal_layer(lay_idx, n_layers, layer, additional_coeffs,
                               lb_k, ub_k, dual_vars):

        # Run get_optim_primal_layer only on subproblem k=lay_idx. Returns the computed partial conditional gradients

        lambdas, rhos = dual_vars.lambdas, dual_vars.rhos
        last_idx = n_layers - 1


        is_first_layer = (lay_idx == 0)
        if is_first_layer:
            lin_km1 = torch.zeros_like(lb_k)
        else:
            lin_km1 = lambdas[lay_idx - 1]
        is_last_layer = (lay_idx == last_idx)
        if is_last_layer:
            lin_hat_k = 0
        else:
            lin_hat_k = -rhos[lay_idx]
        if (lay_idx+1) in additional_coeffs:
            lin_hat_k += additional_coeffs[(lay_idx+1)]
        za_km1, zahat_k = opt_A_domain(layer, lin_km1, lin_hat_k,
                                       lb_k, ub_k, not is_first_layer)

        if is_first_layer:
            return SubproblemCondGrad(lay_idx, None, zahat_k, None, None)

        lin_k = -lambdas[lay_idx - 1]
        lin_hat_k = rhos[lay_idx - 1]

        zbhat_k, zb_k = opt_B_domain(lin_hat_k, lin_k,
                                     lb_k, ub_k)

        return SubproblemCondGrad(lay_idx, za_km1, zahat_k, zb_k, zbhat_k)


    @staticmethod
    def get_initial_kw_solution(weights, additional_coeffs,
                                lower_bounds, upper_bounds):
        assert len(additional_coeffs) > 0
        lambdas = []
        rhos = []

        final_lay_idx = len(weights)
        if final_lay_idx in additional_coeffs:
            # There is a coefficient on the output of the network
            lbda = -weights[final_lay_idx-1].backward(additional_coeffs[final_lay_idx])
            lambdas.append(lbda)
            lay_idx = final_lay_idx
        else:
            # There is none. Just identify the shape from the additional coeffs
            add_coeff = next(iter(additional_coeffs.values()))
            batch_size = add_coeff.shape[0]
            device = lower_bounds[-1].device

            lay_idx = final_lay_idx
            while lay_idx not in additional_coeffs:
                lay_shape = lower_bounds[lay_idx].shape
                if len(lay_shape) == 0:
                    lay_shape = (1,)
                lambdas.append(torch.zeros((batch_size,) + lay_shape,
                                           device=device))
                rhos.append(torch.zeros((batch_size,) + lay_shape,
                                        device=device))
                lay_idx -= 1
            # We now reached the time where lay_idx has an additional coefficient
            rhos.append(torch.zeros_like(additional_coeffs[lay_idx]))
            lbda = -weights[lay_idx-1].backward(additional_coeffs[lay_idx])
            lambdas.append(lbda)

        lay_idx -= 1

        while lay_idx > 0:
            lbs = lower_bounds[lay_idx]
            ubs = upper_bounds[lay_idx]

            scale = ubs / (ubs - lbs)
            scale.masked_fill_(lbs > 0, 1)
            scale.masked_fill_(ubs < 0, 0)

            rho = scale * lbda
            rhos.append(rho)

            lay_idx -= 1
            if lay_idx > 0:
                lay = weights[lay_idx]
                lbda = lay.backward(rho)
                lambdas.append(lbda)

        lambdas.reverse()
        rhos.reverse()

        return DualVarSet(lambdas, rhos)

    @staticmethod
    def get_initial_naive_solution(weights, additional_coeffs,
                                   lower_bounds, upper_bounds):
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[0]
        device = lower_bounds[-1].device

        lambdas = []
        rhos = []
        for lay_idx in range(1, len(weights)):
            lay_shape = lower_bounds[lay_idx].shape
            lambdas.append(torch.zeros((batch_size,) + lay_shape,
                                       device=device))
            rhos.append(torch.zeros((batch_size,) + lay_shape,
                                    device=device))

        return DualVarSet(lambdas, rhos)

    @staticmethod
    def make_primal_full_feasible(dual_vars,
                                  weights, final_coeffs,
                                  lower_bounds, upper_bounds):
        '''
        Return a Primal Var Set, such that there is no gap between the A variables
        and the B variables

        With regards to how to choose the variables, we use the sign of the dual
        variables to guide us if we should be more going towards the top or the
        bottom of the feasible domain.
        '''
        new_zas = []
        new_zahats = []
        new_zbhats = []
        new_zbs = []

        rho_1 = dual_vars.rhos[0]

        lin_eq = weights[0].backward(rho_1)
        lb_km1 = torch.clamp(lower_bounds[0], 0, None)
        ub_km1 = torch.clamp(upper_bounds[0], 0, None)
        pos_coeff = (lin_eq > 0)
        z0 = torch.where(pos_coeff, lb_km1, ub_km1)

        l_1 = lower_bounds[1]
        u_1 = upper_bounds[1]
        zahat = torch.where(rho_1 < 0, u_1, l_1)
        assert (zahat - weights[0].forward(z0)).abs().any() <= 1e-3
        zbhat = zahat
        # Start the propagation
        new_zahats.append(zahat)
        new_zbhats.append(zbhat)

        all_lbdas = dual_vars.lambdas + [final_coeffs]
        lbda1 = all_lbdas[0]
        ambiguous_lb = torch.clamp(zbhat, 0, None)
        ambiguous_ub = (u_1 / (u_1 - l_1)) * (zbhat - l_1)
        ambiguous_val = torch.where(lbda1 < 0, ambiguous_ub, ambiguous_lb)
        zb = torch.where(l_1 > 0,
                         zbhat,
                         torch.where(u_1 < 0,
                                     torch.zeros_like(zbhat),
                                     ambiguous_val))
        za=zb
        new_zbs.append(zb)
        new_zas.append(za)

        for weight, lb, ub, lbda in zip(weights[1:-1], lower_bounds[2:], upper_bounds[2:], all_lbdas[1:]):
            # Compute the next level zahat
            zahat = weight.forward(za)
            zbhat = zahat

            # Compute the value for the relaxation
            ambiguous_lb = torch.clamp(zbhat, 0, None)
            ambiguous_ub = (ub / (ub - lb)) * (zbhat - lb)
            ambiguous_val = torch.where(lbda < 0, ambiguous_ub, ambiguous_lb)
            zb = torch.where(lb > 0,
                             zbhat,
                             torch.where(ub < 0,
                                         torch.zeros_like(zbhat),
                                         ambiguous_val))
            za = zb
            new_zas.append(za)
            new_zahats.append(zahat)
            new_zbhats.append(zbhat)
            new_zbs.append(zb)
        final_zahat = weights[-1].forward(za)
        new_zahats.append(final_zahat)

        assert len(new_zahats) == len(new_zas) + 1
        assert len(new_zahats) == len(new_zbs) + 1
        assert len(new_zahats) == len(new_zbhats) + 1

        primal_feasible = PrimalVarSet(new_zas, new_zahats, new_zbs, new_zbhats, z0)
        return primal_feasible

class PrimalVarSet:
    def __init__(self, zas, zahats, zbs, zbhats, z0):
        self.zas = zas
        self.zahats = zahats
        self.zbs = zbs
        self.zbhats = zbhats
        self.z0 = z0

    def as_dual_subgradient(self):
        lambda_eq = []
        rho_eq = []
        for za, zb in zip(self.zas, self.zbs):
            lambda_eq.append(za - zb)
        for zahat, zbhat in zip(self.zahats, self.zbhats):
            rho_eq.append(zbhat - zahat)
        return DualVarSet(lambda_eq, rho_eq)

    def get_layer_subgradient(self, lay_idx):
        """
        Returns the subgradient for layer lay_idx (as two tensors of shape batch_size x layer width)
        """
        return self.zas[lay_idx] - self.zbs[lay_idx], self.zbhats[lay_idx] - self.zahats[lay_idx]

    def weighted_combination(self, other, coeff):
        new_zas = []
        new_zahats = []
        new_zbs = []
        new_zbhats = []

        # Need to fix how many dim we expand depending on network size
        coeffs = []
        for zahat in self.zahats:
            nb_coeff_expands = (zahat.dim() - 1)
            coeffs.append(coeff.view((coeff.shape[0],) + (1,)*nb_coeff_expands))

        for za, oza, coeffd in zip(self.zas, other.zas, coeffs):
            new_zas.append(za + coeffd * (oza - za))
        for zahat, ozahat, coeffd in zip(self.zahats, other.zahats, coeffs):
            new_zahats.append(zahat + coeffd * (ozahat - zahat))
        for zb, ozb, coeffd in zip(self.zbs, other.zbs, coeffs):
            new_zbs.append(zb + coeffd * (ozb - zb))
        for zbhat, ozbhat, coeffd in zip(self.zbhats, other.zbhats, coeffs):
            new_zbhats.append(zbhat + coeffd * (ozbhat - zbhat))

        coeff0 = coeff.view((coeff.shape[0],) + (1,) * (self.z0.dim() - 1))
        new_z0 = self.z0 + coeff0 * (other.z0 - self.z0)

        return PrimalVarSet(new_zas, new_zahats, new_zbs, new_zbhats, new_z0)

    def weighted_combination_subproblem(self, subproblem, coeff):
        # Perform a weighted combination on the zahats and zbhats that correspond to subproblem k.

        k = subproblem.k
        o_zahat = subproblem.zahat_k
        o_za = subproblem.za_km1
        o_zbhat = subproblem.zbhat_k
        o_zb = subproblem.zb_k

        coeffd_a = coeff.view((coeff.shape[0],) + (1,) * (self.zahats[k].dim() - 1))
        self.zahats[k] = self.zahats[k] + coeffd_a * (o_zahat - self.zahats[k])

        if k > 0:
            coeffd_b = coeff.view((coeff.shape[0],) + (1,) * (self.zbhats[k - 1].dim() - 1))
            self.zas[k - 1] = self.zas[k - 1] + coeffd_b * (o_za - self.zas[k - 1])
            self.zbhats[k - 1] = self.zbhats[k - 1] + coeffd_b * (o_zbhat - self.zbhats[k - 1])
            self.zbs[k - 1] = self.zbs[k - 1] + coeffd_b * (o_zb - self.zbs[k - 1])
        else:
            coeff0 = coeff.view((coeff.shape[0],) + (1,) * (self.z0.dim() - 1))
            self.z0 = self.z0 + coeff0 * (subproblem.z0 - self.z0)

        return self

    def assert_subproblems_feasible(self, weights, final_coeffs,
                                    lower_bounds, upper_bounds):
        # Check all the A subproblems
        for w, zam1, zahat in zip(weights[1:], self.zas, self.zahats[1:]):
            err = w.forward(zam1) - zahat
            assert err.abs().max() == 0
        # Check all the B subproblems
        for lb, ub, zbhat, zb in zip(lower_bounds[1:], upper_bounds[1:],
                                     self.zbhats, self.zbs):
            assert (zb - zbhat).min() >= -1e-6
            assert zb.min() >= 0

            amb_mask = ((lb < 0) & (ub > 0)).type_as(lb)
            assert (amb_mask * ((ub / (ub - lb)) * (zbhat - lb) - zb)).min() >= -1e-6
            pass_mask = (lb > 0).type_as(lb)
            assert (pass_mask * (zb - zbhat)).abs().max() == 0
            block_mask = (ub < 0).type_as(ub)
            assert (block_mask * zb).abs().max() == 0
            assert (block_mask * zbhat).max() == 0

class DualVarSet:
    def __init__(self, lambdas, rhos):
        self.lambdas = lambdas
        self.rhos = rhos

    def bdot(self, other):
        val = 0
        for lbda, olbda in zip(self.lambdas, other.lambdas):
            val += bdot(lbda, olbda)
        for rho, orho in zip(self.rhos, other.rhos):
            val += bdot(rho, orho)
        return val

    def add_(self, step_size, to_add):
        for lbda, lbda_step in zip(self.lambdas, to_add.lambdas):
            lbda.add_(step_size, lbda_step)
        for rho, rho_step in zip(self.rhos, to_add.rhos):
            rho.add_(step_size, rho_step)
        return self

    def add_cte_(self, cte):
        for lbda in self.lambdas:
            lbda.add_(cte)
        for rho in self.rhos:
            rho.add_(cte)
        return self

    def addcmul_(self, coeff, to_add1, to_add2):
        for lbda, lbda1, lbda2 in zip(self.lambdas, to_add1.lambdas, to_add2.lambdas):
            lbda.addcmul_(coeff, lbda1, lbda2)
        for rho, rho1, rho2 in zip(self.rhos, to_add1.rhos, to_add2.rhos):
            rho.addcmul_(coeff, rho1, rho2)
        return self

    def addcdiv_(self, coeff, num, denom):
        for lbda, num_lbda, denom_lbda in zip(self.lambdas, num.lambdas, denom.lambdas):
            lbda.addcdiv_(coeff, num_lbda, denom_lbda)
        for rho, num_rho, denom_rho in zip(self.rhos, num.rhos, denom.rhos):
            rho.addcdiv_(coeff, num_rho, denom_rho)
        return self

    def mul_(self, coeff):
        for lbda in self.lambdas:
            lbda.mul_(coeff)
        for rho in self.rhos:
            rho.mul_(coeff)
        return self

    def zero_like(self):
        new_lambdas = []
        new_rhos = []
        for lbda in self.lambdas:
            new_lambdas.append(torch.zeros_like(lbda))
        for rho in self.rhos:
            new_rhos.append(torch.zeros_like(rho))
        return DualVarSet(new_lambdas, new_rhos)

    def add(self, to_add, step_size):
        new_lambdas = []
        new_rhos = []
        for lbda, lbda_step in zip(self.lambdas, to_add.lambdas):
            new_lambdas.append(lbda + step_size * lbda_step)
        for rho, rho_step in zip(self.rhos, to_add.rhos):
            new_rhos.append(rho + step_size * rho_step)
        return DualVarSet(new_lambdas, new_rhos)

    def sqrt(self):
        new_lambdas = [lbda.sqrt() for lbda in self.lambdas]
        new_rhos = [rho.sqrt() for rho in self.rhos]
        return DualVarSet(new_lambdas, new_rhos)

    def clone(self):
        new_lambdas = [l.clone() for l in self.lambdas]
        new_rhos = [r.clone() for r in self.rhos]
        return DualVarSet(new_lambdas, new_rhos)

    def weighted_squared_norm(self, eta):
        val = 0
        batch_size = self.rhos[0].shape[0]
        for lbda in self.lambdas:
            val += eta * lbda.view(batch_size, -1).pow(2).sum(dim=-1)
        for rho in self.rhos:
            val += eta * rho.view(batch_size, -1).pow(2).sum(dim=-1)
        return val

    def assert_zero(self):
        for lbda in self.lambdas:
            assert lbda.abs().max() == 0
        for rho in self.rhos:
            assert rho.abs().max() == 0

    def update_from_anchor_points(self, anchor_point, primal_vars, eta, lay_idx="all"):
        """
        Given the anchor point (DualVarSet instance) and primal vars (as PrimalVarSet instance), compute and return the
        updated dual variables (anchor points) with their
        closed-form from KKT conditions. The update is performed in place.
         lay_idx are the layers (int or list) for which to perform the update. "all" means update all
        """
        if lay_idx == "all":
            lay_to_iter = range(len(self.rhos))
        else:
            lay_to_iter = [lay_idx] if type(lay_idx) is int else list(lay_idx)

        for lay_idx in lay_to_iter:
            lambda_update, rho_update = primal_vars.get_layer_subgradient(lay_idx)
            self.lambdas[lay_idx] = anchor_point.lambdas[lay_idx] + (1 / eta) * lambda_update
            self.rhos[lay_idx] = anchor_point.rhos[lay_idx] + (1 / eta) * rho_update

    def copy(self):
        """
        deep-copy the current instance
        :return: the copied class instance
        """
        return DualVarSet(copy.deepcopy(self.lambdas), copy.deepcopy(self.rhos))


class SubproblemCondGrad:
    # Contains the variables corresponding to a single subproblem conditional gradient computation
    def __init__(self, k, za_km1, zahat_k, zb_k, zbhat_k, z0=None):
        self.k = k
        self.za_km1 = za_km1
        self.zahat_k = zahat_k
        self.zb_k = zb_k
        self.zbhat_k = zbhat_k
        self.z0 = z0  # non-None only for the first layer

    def proximal_optimal_step_size_subproblem(self, additional_coeffs, diff_grad, primal_vars, n_layers, eta):
        # Compute proximal_optimal_step_size knowing that only the conditional
        # gradient of subproblem k was updated.

        k = self.k
        za_km1 = self.za_km1
        zahat_k = self.zahat_k
        zb_k = self.zb_k
        zbhat_k = self.zbhat_k

        ahat_diff = zahat_k - primal_vars.zahats[k]

        if k == 0:
            upper = bdot(diff_grad.rhos[0], ahat_diff)
            lower = (1 / eta) * ahat_diff.view(ahat_diff.shape[0], -1).pow(2).sum(dim=-1)
        else:
            a_diff = primal_vars.zas[k-1] - za_km1
            b_diff = zb_k - primal_vars.zbs[k - 1]
            upper = bdot(diff_grad.lambdas[k - 1], a_diff + b_diff)
            bhat_diff = primal_vars.zbhats[k - 1]- zbhat_k
            upper += bdot(diff_grad.rhos[k - 1], bhat_diff)

            low_diff = b_diff + a_diff
            lower = (1 / eta) * low_diff.view(low_diff.shape[0], -1).pow(2).sum(dim=-1)
            lower += (1 / eta) * bhat_diff.view(bhat_diff.shape[0], -1).pow(2).sum(dim=-1)

            if k != (n_layers-1):
                upper += bdot(diff_grad.rhos[k], ahat_diff)
                lower += (1 / eta) * ahat_diff.view(ahat_diff.shape[0], -1).pow(2).sum(dim=-1)
            if (k+1) in additional_coeffs:
                upper += bdot(additional_coeffs[k+1], primal_vars.zahats[-1] - zahat_k)

        opt_step_size = torch.where(lower > 0, upper / lower, torch.zeros_like(lower))
        # Set to 0 the 0/0 entries.
        up_mask = upper == 0
        low_mask = lower == 0
        sum_mask = up_mask + low_mask
        opt_step_size[sum_mask > 1] = 0
        opt_step_size = torch.clamp(opt_step_size, min=0, max=1)

        decrease = -0.5 * lower * opt_step_size.pow(2) + upper * opt_step_size

        return opt_step_size, decrease


def opt_A_domain(layer,
                 lin_km1, lin_hat_k,
                 lb_km1, ub_km1, relu_bounds):
    '''
    Relu_bounds indicates whether the bounds should be ReLUified.
    Always yes, except for the first layer.
    '''
    if lin_hat_k is 0:
        lin_eq = lin_km1
    else:
        lin_eq = lin_km1 + layer.backward(lin_hat_k)

    if relu_bounds:
        lb_km1 = torch.clamp(lb_km1, 0, None)
        ub_km1 = torch.clamp(ub_km1, 0, None)

    pos_coeff = (lin_eq > 0)
    za_km1 = torch.where(pos_coeff, lb_km1, ub_km1)
    zahat_k = layer.forward(za_km1)

    return za_km1, zahat_k


def opt_B_domain(lin_hat_k, lin_k, lb_k, ub_k):
    rel_lb = torch.clamp(lb_k, 0, None)
    rel_ub = torch.clamp(ub_k, 0, None)

    zero_vertex = torch.zeros_like(lb_k)
    unambiguous = (ub_k < 0) | (lb_k > 0)
    zero_vertex.masked_fill_(unambiguous, float('inf'))
    vertex_vals = torch.stack((
        lin_hat_k * lb_k + lin_k * rel_lb,
        lin_hat_k * ub_k + lin_k * rel_ub,
        zero_vertex.unsqueeze(0).expand_as(lin_hat_k)
    ))
    _, choice = torch.min(vertex_vals, 0)

    zbhat_k = torch.where(choice == 0, lb_k, ub_k)
    zbhat_k.masked_fill_(choice == 2, 0)

    zb_k = torch.where(choice == 0, rel_lb, rel_ub)
    zb_k.masked_fill_(choice == 2, 0)

    return zbhat_k, zb_k
