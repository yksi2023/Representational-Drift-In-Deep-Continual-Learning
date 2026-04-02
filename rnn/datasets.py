import torch
import numpy as np

# Rule definitions from Yang et al.
rules_dict = {
    'all': ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
            'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
            'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
            'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
    'mante': ['contextdm1', 'contextdm2'],
    'oicdmc': ['oic', 'dmc']
}

rule_index_map = dict()
for ruleset, rules in rules_dict.items():
    rule_index_map[ruleset] = dict()
    for ind, rule in enumerate(rules):
        rule_index_map[ruleset][rule] = ind

def get_rule_index(rule, config):
    '''get the input index for the given rule'''
    return rule_index_map[config['ruleset']][rule] + config['rule_start']

def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi - abs(original_dist))

class Trial(object):
    """Class representing a batch of trials. Ported from Yang et al."""

    def __init__(self, config, tdim, batch_size):
        self.float_type = 'float32'
        self.config = config
        self.dt = self.config.get('dt', 20)

        self.n_eachring = self.config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        self.pref = np.arange(0, 2*np.pi, 2*np.pi/self.n_eachring)

        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)
        
        if self.config.get('loss_type', 'cross_entropy') == 'lsq':
            self.y[:, :, :] = 0.05
            
        self.y_loc = -np.ones((tdim, batch_size), dtype=self.float_type)
        self._sigma_x = config.get('sigma_x', 0.01) * np.sqrt(2 / config.get('alpha', 0.2))
        self.epochs = {}

    def expand(self, var):
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]: offs[i], i, 0] = 1
            elif loc_type == 'stim':
                self.x[ons[i]: offs[i], i, 1+(mods[i]-1)*self.n_eachring : 1+mods[i]*self.n_eachring] \
                    += self.add_x_loc(locs[i]) * strengths[i]
            elif loc_type == 'fix_out':
                if self.config.get('loss_type', 'cross_entropy') == 'lsq':
                    self.y[ons[i]: offs[i], i, 0] = 0.8
                else:
                    self.y[ons[i]: offs[i], i, 0] = 1.0
            elif loc_type == 'out':
                if self.config.get('loss_type', 'cross_entropy') == 'lsq':
                    self.y[ons[i]: offs[i], i, 1:] += self.add_y_loc(locs[i]) * strengths[i]
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    if np.sum(y_tmp) > 0:
                        y_tmp /= np.sum(y_tmp)
                    self.y[ons[i]: offs[i], i, 1:] += y_tmp
                self.y_loc[ons[i]: offs[i], i] = locs[i]
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        rng = self.config.get('rng', np.random.RandomState())
        self.x += rng.randn(*self.x.shape) * self._sigma_x

    def add_c_mask(self, pre_offs, post_ons):
        pre_on = int(100 / self.dt)
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)

        if self.config.get('loss_type', 'cross_entropy') == 'lsq':
            c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
            for i in range(self.batch_size):
                c_mask[post_ons[i]:, i, :] = 5.
                c_mask[pre_on:pre_offs[i], i, :] = 1.
            c_mask[:, :, 0] *= 2.
            self.c_mask = c_mask
        else:
            c_mask = np.zeros((self.tdim, self.batch_size), dtype=self.float_type)
            for i in range(self.batch_size):
                if post_ons[i] is not None:
                    c_mask[post_ons[i]:, i] = 5.
                if pre_offs[i] is not None:
                    c_mask[pre_on:pre_offs[i], i] = 1.
            self.c_mask = c_mask / (c_mask.mean() + 1e-8)

    def add_rule(self, rule, on=None, off=None, strength=1.):
        if isinstance(rule, int):
            self.x[on:off, :, self.config['rule_start'] + rule] = strength
        else:
            ind_rule = get_rule_index(rule, self.config)
            self.x[on:off, :, ind_rule] = strength

    def add_x_loc(self, x_loc):
        dist = get_dist(x_loc - self.pref)
        dist /= np.pi/8
        return 0.8 * np.exp(-dist**2 / 2)

    def add_y_loc(self, y_loc):
        dist = get_dist(y_loc - self.pref)
        if self.config.get('loss_type', 'cross_entropy') == 'lsq':
            dist /= np.pi/8
            y = 0.8 * np.exp(-dist**2 / 2)
        else:
            y = np.zeros_like(dist)
            ind = np.argmin(dist)
            y[ind] = 1.
        return y
        
    def to_tensor(self, device='cpu'):
        """Converts trial data to PyTorch tensors. Returns cached GPU tensors if available."""
        if hasattr(self, '_cached_tensors'):
            return self._cached_tensors
        x_tensor = torch.tensor(self.x, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(self.y, dtype=torch.float32, device=device)
        c_mask_tensor = torch.tensor(self.c_mask, dtype=torch.float32, device=device)
        return x_tensor, y_tensor, c_mask_tensor

    def cache_to_device(self, device):
        """Pre-convert numpy arrays to GPU tensors and cache them."""
        self._cached_tensors = (
            torch.tensor(self.x, dtype=torch.float32, device=device),
            torch.tensor(self.y, dtype=torch.float32, device=device),
            torch.tensor(self.c_mask, dtype=torch.float32, device=device),
        )


# ----------------------------------------------------------------------
# Helper for context DM stimulus generation
# ----------------------------------------------------------------------

def _contextdm_genstim(batch_size, rng, stim_coh_range=None):
    """Generate stimulus strengths for context-dependent DM tasks."""
    stim_mean = rng.uniform(0.8, 1.2, (batch_size,))
    if stim_coh_range is None:
        stim_coh_range = np.array([0.16, 0.32, 0.64])
    stim_coh = rng.choice(stim_coh_range, (batch_size,))
    stim_sign = rng.choice([+1, -1], (batch_size,))
    stim1_strengths = stim_mean + stim_coh * stim_sign
    stim2_strengths = stim_mean - stim_coh * stim_sign
    return stim1_strengths, stim2_strengths


# ----------------------------------------------------------------------
# Wrapper: adds rule input and noise to any raw task generator
# ----------------------------------------------------------------------

def _finalize_trial(trial, rule_name, config):
    """Add rule input and input noise to a trial."""
    trial.add_rule(rule_name)
    trial.add_x_noise()
    return trial


# ======================================================================
# Go Family (6 tasks)
# ======================================================================

def _delaygo(config, batch_size, mode='random', anti_response=False, **kwargs):
    """Delay-Go / Delay-Anti: fixate, observe stimulus, delay, then saccade."""
    dt = config['dt']
    rng = config['rng']
    if mode != 'random':
        raise ValueError(f'Unsupported mode: {mode}')

    stim_locs = rng.rand(batch_size) * 2 * np.pi
    stim_ons = int(rng.choice([300, 500, 700]) / dt)
    stim_offs = stim_ons + int(rng.choice([200, 400, 600]) / dt)
    fix_offs = stim_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    tdim = fix_offs + int(500 / dt)
    stim_mod = rng.choice([1, 2])

    check_ons = fix_offs + int(100 / dt)
    stim_locs = np.array(stim_locs)
    response_locs = stim_locs if not anti_response else (stim_locs + np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim_ons), 'stim1': (stim_ons, stim_offs),
                    'delay1': (stim_offs, fix_offs), 'go1': (fix_offs, None)}
    return trial


def _fdgo(config, batch_size, mode='random', anti_response=False, **kwargs):
    """Fixed-Delay Go / Anti: stimulus shown from onset until go, no blank delay."""
    dt = config['dt']
    rng = config['rng']
    if mode != 'random':
        raise ValueError(f'Unsupported mode: {mode}')

    stim_locs = rng.rand(batch_size) * 2 * np.pi
    stim_mod = rng.choice([1, 2])
    stim_ons = int(rng.uniform(300, 700) / dt)
    fix_offs = stim_ons + int(rng.uniform(500, 1500) / dt)
    tdim = int(500 / dt) + fix_offs

    check_ons = fix_offs + int(100 / dt)
    stim_locs = np.array(stim_locs)
    response_locs = stim_locs if not anti_response else (stim_locs + np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim_ons), 'stim1': (stim_ons, fix_offs),
                    'go1': (fix_offs, None)}
    return trial


def _reactgo(config, batch_size, mode='random', anti_response=False, **kwargs):
    """React Go / Anti: fixate until stimulus appears, then immediately saccade."""
    dt = config['dt']
    rng = config['rng']
    if mode != 'random':
        raise ValueError(f'Unsupported mode: {mode}')

    stim_ons = int(rng.uniform(500, 2500) / dt)
    tdim = int(500 / dt) + stim_ons
    stim_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim_mod = rng.choice([1, 2])

    check_ons = stim_ons + int(100 / dt)
    stim_locs = np.array(stim_locs)
    response_locs = stim_locs if not anti_response else (stim_locs + np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in')
    trial.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
    trial.add('fix_out', offs=stim_ons)
    trial.add('out', response_locs, ons=stim_ons)
    trial.add_c_mask(pre_offs=stim_ons, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim_ons), 'go1': (stim_ons, None)}
    return trial


def generate_delaygo(config, batch_size, mode='random', **kwargs):
    trial = _delaygo(config, batch_size, mode, anti_response=False, **kwargs)
    return _finalize_trial(trial, 'delaygo', config)

def generate_delayanti(config, batch_size, mode='random', **kwargs):
    trial = _delaygo(config, batch_size, mode, anti_response=True, **kwargs)
    return _finalize_trial(trial, 'delayanti', config)

def generate_fdgo(config, batch_size, mode='random', **kwargs):
    trial = _fdgo(config, batch_size, mode, anti_response=False, **kwargs)
    return _finalize_trial(trial, 'fdgo', config)

def generate_fdanti(config, batch_size, mode='random', **kwargs):
    trial = _fdgo(config, batch_size, mode, anti_response=True, **kwargs)
    return _finalize_trial(trial, 'fdanti', config)

def generate_reactgo(config, batch_size, mode='random', **kwargs):
    trial = _reactgo(config, batch_size, mode, anti_response=False, **kwargs)
    return _finalize_trial(trial, 'reactgo', config)

def generate_reactanti(config, batch_size, mode='random', **kwargs):
    trial = _reactgo(config, batch_size, mode, anti_response=True, **kwargs)
    return _finalize_trial(trial, 'reactanti', config)


# ======================================================================
# Decision-Making Family (5 tasks)
# ======================================================================

def _dm(config, batch_size, mode='random', stim_mod=1, **kwargs):
    """DM: two stimuli shown simultaneously, saccade to the stronger one."""
    dt = config['dt']
    rng = config['rng']
    if mode != 'random':
        raise ValueError(f'Unsupported mode: {mode}')

    stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

    stims_mean = rng.uniform(0.8, 1.2, (batch_size,))
    stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
    if config.get('easy_task', False):
        stim_coh_range *= 10
    stims_coh = rng.choice(stim_coh_range, (batch_size,))
    stims_sign = rng.choice([1, -1], (batch_size,))
    stim1_strengths = stims_mean + stims_coh * stims_sign
    stim2_strengths = stims_mean - stims_coh * stims_sign

    stim_on = int(rng.uniform(100, 400) / dt)
    stim_ons = (np.ones(batch_size) * stim_on).astype(int)
    stim_dur = int(rng.choice([400, 800, 1600]) / dt)
    fix_offs = (stim_ons + stim_dur).astype(int)
    tdim = stim_on + stim_dur + int(500 / dt)
    check_ons = fix_offs + int(100 / dt)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=fix_offs, strengths=stim1_strengths, mods=stim_mod)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=fix_offs, strengths=stim2_strengths, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if stim1_strengths[i] > stim2_strengths[i]
                 else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim_ons), 'stim1': (stim_ons, fix_offs),
                    'go1': (fix_offs, None)}
    return trial


def _contextdm(config, batch_size, mode='random', attend_mod=1, **kwargs):
    """Context DM: two stimuli in each modality ring, attend to one modality."""
    dt = config['dt']
    rng = config['rng']
    if mode != 'random':
        raise ValueError(f'Unsupported mode: {mode}')

    stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

    stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
    if config.get('easy_task', False):
        stim_coh_range *= 10

    if attend_mod in (1, 2):
        stim1_mod1_strengths, stim2_mod1_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
        stim1_mod2_strengths, stim2_mod2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
        if attend_mod == 1:
            stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
        else:
            stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    else:  # attend_mod == 'both' (multidm)
        stim1_strengths, stim2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
        stim1_mod12_diff = stim1_strengths * rng.uniform(0.2, 0.8, (batch_size,)) * rng.choice([+1, -1], (batch_size,))
        stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff / 2
        stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff / 2
        stim2_mod12_diff = stim2_strengths * rng.uniform(0.2, 0.8, (batch_size,)) * rng.choice([+1, -1], (batch_size,))
        stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff / 2
        stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff / 2

    stim_on = int(rng.uniform(100, 400) / dt)
    stim_ons = (np.ones(batch_size) * stim_on).astype(int)
    stim_dur = int(rng.choice([400, 800, 1600]) / dt)
    stim_offs = stim_ons + stim_dur
    fix_offs = stim_offs  # No delay for context DM
    tdim = stim_on + stim_dur + int(500 / dt)
    check_ons = fix_offs + int(100 / dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == 'both':
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=stim_offs, strengths=stim1_mod1_strengths, mods=1)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=stim_offs, strengths=stim2_mod1_strengths, mods=1)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=stim_offs, strengths=stim1_mod2_strengths, mods=2)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=stim_offs, strengths=stim2_mod2_strengths, mods=2)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if stim1_strengths[i] > stim2_strengths[i]
                 else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim_ons), 'stim1': (stim_ons, stim_offs),
                    'go1': (fix_offs, None)}
    return trial


def generate_dm1(config, batch_size, mode='random', **kwargs):
    trial = _dm(config, batch_size, mode, stim_mod=1, **kwargs)
    return _finalize_trial(trial, 'dm1', config)

def generate_dm2(config, batch_size, mode='random', **kwargs):
    trial = _dm(config, batch_size, mode, stim_mod=2, **kwargs)
    return _finalize_trial(trial, 'dm2', config)

def generate_contextdm1(config, batch_size, mode='random', **kwargs):
    trial = _contextdm(config, batch_size, mode, attend_mod=1, **kwargs)
    return _finalize_trial(trial, 'contextdm1', config)

def generate_contextdm2(config, batch_size, mode='random', **kwargs):
    trial = _contextdm(config, batch_size, mode, attend_mod=2, **kwargs)
    return _finalize_trial(trial, 'contextdm2', config)

def generate_multidm(config, batch_size, mode='random', **kwargs):
    trial = _contextdm(config, batch_size, mode, attend_mod='both', **kwargs)
    return _finalize_trial(trial, 'multidm', config)


# ======================================================================
# Delay Decision-Making Family (5 tasks)
# ======================================================================

def _delaydm(config, batch_size, mode='random', stim_mod=1, **kwargs):
    """Delay DM: two stimuli shown sequentially, saccade to the stronger one."""
    dt = config['dt']
    rng = config['rng']
    if mode != 'random':
        raise ValueError(f'Unsupported mode: {mode}')

    stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

    stims_mean = rng.uniform(0.8, 1.2, (batch_size,))
    stim_coh_range = np.array([0.08, 0.16, 0.32])
    if config.get('easy_task', False):
        stim_coh_range *= 2
    stims_coh = rng.choice(stim_coh_range, (batch_size,))
    stims_sign = rng.choice([1, -1], (batch_size,))
    stim1_strengths = stims_mean + stims_coh * stims_sign
    stim2_strengths = stims_mean - stims_coh * stims_sign

    stim1_ons = int(rng.choice([200, 400, 600]) / dt)
    stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
    stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    stim2_offs = stim2_ons + int(rng.choice([200, 400, 600]) / dt)
    fix_offs = stim2_offs + int(rng.uniform(100, 300) / dt)
    tdim = fix_offs + int(500 / dt)
    check_ons = fix_offs + int(100 / dt)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_strengths, mods=stim_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_strengths, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if stim1_strengths[i] > stim2_strengths[i]
                 else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim1_ons), 'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons), 'stim2': (stim2_ons, stim2_offs),
                    'delay2': (stim2_offs, fix_offs), 'go1': (fix_offs, None)}
    return trial


def _contextdelaydm(config, batch_size, mode='random', attend_mod=1, **kwargs):
    """Context Delay DM: two sequential stimuli in each modality, attend to one."""
    dt = config['dt']
    rng = config['rng']
    if mode != 'random':
        raise ValueError(f'Unsupported mode: {mode}')

    stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

    stim_coh_range = np.array([0.08, 0.16, 0.32])
    if config.get('easy_task', False):
        stim_coh_range *= 2

    if attend_mod in (1, 2):
        stim1_mod1_strengths, stim2_mod1_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
        stim1_mod2_strengths, stim2_mod2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
        if attend_mod == 1:
            stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
        else:
            stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    else:  # 'both'
        stim1_strengths, stim2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
        stim1_mod12_diff = stim1_strengths * rng.uniform(0.2, 0.8, (batch_size,)) * rng.choice([+1, -1], (batch_size,))
        stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff / 2
        stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff / 2
        stim2_mod12_diff = stim2_strengths * rng.uniform(0.2, 0.8, (batch_size,)) * rng.choice([+1, -1], (batch_size,))
        stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff / 2
        stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff / 2

    stim1_ons = int(rng.choice([200, 400, 600]) / dt)
    stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
    stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    stim2_offs = stim2_ons + int(rng.choice([200, 400, 600]) / dt)
    fix_offs = stim2_offs + int(rng.uniform(100, 300) / dt)
    tdim = fix_offs + int(500 / dt)
    check_ons = fix_offs + int(100 / dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == 'both':
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_mod1_strengths, mods=1)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_mod1_strengths, mods=1)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_mod2_strengths, mods=2)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_mod2_strengths, mods=2)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if stim1_strengths[i] > stim2_strengths[i]
                 else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim1_ons), 'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons), 'stim2': (stim2_ons, stim2_offs),
                    'delay2': (stim2_offs, fix_offs), 'go1': (fix_offs, None)}
    return trial


def generate_delaydm1(config, batch_size, mode='random', **kwargs):
    trial = _delaydm(config, batch_size, mode, stim_mod=1, **kwargs)
    return _finalize_trial(trial, 'delaydm1', config)

def generate_delaydm2(config, batch_size, mode='random', **kwargs):
    trial = _delaydm(config, batch_size, mode, stim_mod=2, **kwargs)
    return _finalize_trial(trial, 'delaydm2', config)

def generate_contextdelaydm1(config, batch_size, mode='random', **kwargs):
    trial = _contextdelaydm(config, batch_size, mode, attend_mod=1, **kwargs)
    return _finalize_trial(trial, 'contextdelaydm1', config)

def generate_contextdelaydm2(config, batch_size, mode='random', **kwargs):
    trial = _contextdelaydm(config, batch_size, mode, attend_mod=2, **kwargs)
    return _finalize_trial(trial, 'contextdelaydm2', config)

def generate_multidelaydm(config, batch_size, mode='random', **kwargs):
    trial = _contextdelaydm(config, batch_size, mode, attend_mod='both', **kwargs)
    return _finalize_trial(trial, 'multidelaydm', config)


# ======================================================================
# Match Family (4 tasks)
# ======================================================================

def _dms(config, batch_size, mode='random', matchnogo=0, **kwargs):
    """Delay-Match-to-Sample: match=go or match=nogo depending on matchnogo."""
    dt = config['dt']
    rng = config['rng']
    if mode != 'random':
        raise ValueError(f'Unsupported mode: {mode}')

    stim1_mod = rng.choice([1, 2])
    stim2_mod = rng.choice([1, 2])
    matchs = rng.choice([0, 1], (batch_size,))
    stim_dist = rng.uniform(np.pi / 9, np.pi * 17. / 9., (batch_size,)) * rng.choice([-1, 1], (batch_size,))
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim2_locs = (stim1_locs + stim_dist * (1 - matchs)) % (2 * np.pi)

    stim1_ons = int(rng.choice([200, 400, 600]) / dt)
    stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
    stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    tdim = stim2_ons + int(500 / dt)
    check_ons = stim2_ons + int(100 / dt)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in')
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, mods=stim2_mod)

    fix_out_offs = [stim2_ons] * batch_size
    out_offs = [None] * batch_size
    for i in range(batch_size):
        if matchs[i] == matchnogo:
            fix_out_offs[i] = None
            out_offs[i] = 0

    trial.add('fix_out', offs=fix_out_offs)
    trial.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)
    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim1_ons), 'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons), 'go1': (stim2_ons, None)}
    return trial


def _dmc(config, batch_size, mode='random', matchnogo=0, **kwargs):
    """Delay-Match-to-Category: category match=go or match=nogo."""
    dt = config['dt']
    rng = config['rng']
    if mode != 'random':
        raise ValueError(f'Unsupported mode: {mode}')

    stim1_mod = rng.choice([1, 2])
    stim2_mod = rng.choice([1, 2])
    stim1_locs = rng.choice(
        np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]) * np.pi, size=(batch_size,))
    stim2_locs = rng.choice(
        np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]) * np.pi, size=(batch_size,))

    stim1_ons = int(rng.choice([200, 400, 600]) / dt)
    stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
    stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    tdim = stim2_ons + int(rng.choice([200, 400, 600]) / dt)
    check_ons = stim2_ons + int(100 / dt)

    stim1_cats = stim1_locs < np.pi
    stim2_cats = stim2_locs < np.pi
    matchs = (stim1_cats == stim2_cats)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in')
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, mods=stim2_mod)

    fix_out_offs = [stim2_ons] * batch_size
    out_offs = [None] * batch_size
    for i in range(batch_size):
        if matchs[i] == matchnogo:
            fix_out_offs[i] = None
            out_offs[i] = 0

    trial.add('fix_out', offs=fix_out_offs)
    trial.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)
    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim1_ons), 'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons), 'go1': (stim2_ons, None)}
    return trial


def generate_dmsgo(config, batch_size, mode='random', **kwargs):
    trial = _dms(config, batch_size, mode, matchnogo=0, **kwargs)
    return _finalize_trial(trial, 'dmsgo', config)

def generate_dmsnogo(config, batch_size, mode='random', **kwargs):
    trial = _dms(config, batch_size, mode, matchnogo=1, **kwargs)
    return _finalize_trial(trial, 'dmsnogo', config)

def generate_dmcgo(config, batch_size, mode='random', **kwargs):
    trial = _dmc(config, batch_size, mode, matchnogo=0, **kwargs)
    return _finalize_trial(trial, 'dmcgo', config)

def generate_dmcnogo(config, batch_size, mode='random', **kwargs):
    trial = _dmc(config, batch_size, mode, matchnogo=1, **kwargs)
    return _finalize_trial(trial, 'dmcnogo', config)


# ======================================================================
# Task Registry
# ======================================================================

TASK_REGISTRY = {
    'fdgo': generate_fdgo,
    'reactgo': generate_reactgo,
    'delaygo': generate_delaygo,
    'fdanti': generate_fdanti,
    'reactanti': generate_reactanti,
    'delayanti': generate_delayanti,
    'dm1': generate_dm1,
    'dm2': generate_dm2,
    'contextdm1': generate_contextdm1,
    'contextdm2': generate_contextdm2,
    'multidm': generate_multidm,
    'delaydm1': generate_delaydm1,
    'delaydm2': generate_delaydm2,
    'contextdelaydm1': generate_contextdelaydm1,
    'contextdelaydm2': generate_contextdelaydm2,
    'multidelaydm': generate_multidelaydm,
    'dmsgo': generate_dmsgo,
    'dmsnogo': generate_dmsnogo,
    'dmcgo': generate_dmcgo,
    'dmcnogo': generate_dmcnogo,
}

ALL_TASKS = list(TASK_REGISTRY.keys())

# Default task list excludes DMC family (fully discrete stimulus locations cause train/test overlap)
DEFAULT_TASKS = [t for t in ALL_TASKS if t not in ('dmcgo', 'dmcnogo')]


def get_task_generator(task_name):
    """Look up a task generator function by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_name}'. Available: {ALL_TASKS}")
    return TASK_REGISTRY[task_name]


def get_default_config():
    """Returns a default configuration for the tasks."""
    return {
        'dt': 20,
        'alpha': 0.2,
        'rng': np.random.RandomState(),
        'ruleset': 'all',
        'n_eachring': 32,
        'n_input': 1 + 32 * 2 + 20,   # 1 fix + 2 rings of 32 + 20 rules
        'n_output': 1 + 32,            # 1 fix + 1 ring of 32
        'rule_start': 1 + 32 * 2,
        'loss_type': 'cross_entropy',
        'sigma_x': 0.01,
    }
