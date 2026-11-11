"""Matched cross-view continuations from the public AugReg ViT-B/32 weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from safetensors.torch import load_file
from torchvision.models import vit_b_32

import contrastive_benefit_cpu as contrastive
import late_vision_cpu as vision
import vision_auxiliary_cpu as base
from consumer_contracts import evaluate_contracts
from suffix_serving import SuffixReadout, WholeNetwork
from token_suffix import TokenPrefix, TokenSuffix

CHECKPOINT_SHA = 'b0a4866d4f722434e9d26c529181f1e7072d80edcf75843f2af4e091617a4098'


def load_augreg(path):
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == CHECKPOINT_SHA
    weights = load_file(str(path), device='cpu')
    model = vit_b_32(weights=None)
    mapping = {'conv_proj': 'patch_embed.proj', 'encoder.ln': 'norm', 'heads.head': 'head'}
    for index in range(12):
        target = f'encoder.layers.encoder_layer_{index}'
        source = f'blocks.{index}'
        for a, b in {'ln_1': 'norm1', 'ln_2': 'norm2',
                     'self_attention.out_proj': 'attn.proj',
                     'mlp.0': 'mlp.fc1', 'mlp.3': 'mlp.fc2'}.items():
            mapping[f'{target}.{a}'] = f'{source}.{b}'
    state = {f'{target}.{suffix}': weights[f'{source}.{suffix}']
             for target, source in mapping.items() for suffix in ('weight', 'bias')}
    for index in range(12):
        for suffix in ('weight', 'bias'):
            state[f'encoder.layers.encoder_layer_{index}.self_attention.in_proj_{suffix}'] = (
                weights[f'blocks.{index}.attn.qkv.{suffix}'])
    state['class_token'] = weights['cls_token']
    state['encoder.pos_embedding'] = weights['pos_embed']
    assert len(state) == len(weights)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def make_cache(prefix, healthy, images, augmented, fit_and_val, y, coarse):
    tokens = torch.empty(2, len(fit_and_val), 50, 768)
    features = torch.empty(2, len(fit_and_val), 768)
    hidden = torch.empty(2, len(fit_and_val), 3072)
    for view, source in enumerate((images, augmented)):
        for start in range(0, len(fit_and_val), 128):
            indices = fit_and_val[start:start + 128]
            section = slice(start, start + len(indices))
            x = prefix(vision.preprocess(source[indices], 224))
            h, r = healthy.features(x)
            tokens[view, section], features[view, section], hidden[view, section] = x, h, r
            if start % 4096 == 0:
                print(json.dumps(dict(view=view, prepared=start)), flush=True)
    return dict(x=tokens, h=features, hidden=hidden, y=y[fit_and_val],
                coarse=coarse[fit_and_val])


def main(args):
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    model = load_augreg(args.checkpoint)
    layers = list(model.encoder.layers)
    prefix = TokenPrefix(model, 2).eval()
    healthy = TokenSuffix(layers[-1], model.encoder.ln, layers[-2:-1]).eval()
    healthy.requires_grad_(False)
    images, y, coarse, data_sha = vision.images(True)
    augmented, augmentation = base.augment(images)
    order = torch.randperm(50000, generator=torch.Generator().manual_seed(args.reference_seed))
    mask = torch.rand(10000, generator=torch.Generator().manual_seed(args.reference_seed + 10000)) < .2
    with torch.no_grad():
        sample = vision.preprocess(images[order[:128]], 224)
        tokens = torch.cat([model.class_token.expand(len(sample), -1, -1),
                            model._process_input(sample)], dim=1)
        torch.testing.assert_close(healthy(prefix(sample)), model.encoder(tokens)[:, 0],
                                   atol=3e-5, rtol=3e-5)
    cache = make_cache(prefix, healthy, images, augmented, order[:15000], y, coarse)
    torch.manual_seed(args.reference_seed)
    ref = base.fit_reference(cache, args.reference_seed, torch.arange(10000),
                             torch.arange(10000, 15000), torch.empty(0, dtype=torch.long), mask)
    outcomes = {}
    for name, alpha in [('supervised', 0.), ('contrastive', .1)]:
        settings = SimpleNamespace(alpha=alpha, epochs=10, primary_two_view=False,
                                   seed=6310, reference_seed=args.reference_seed)
        trained, head, reader, logits, _, metrics, auxiliary, trace = contrastive.train(
            cache, ref, healthy, settings)
        outcomes[name] = dict(accuracy=metrics['acc'], loss=auxiliary, trace=trace)
        if name == 'supervised':
            baseline_logits = logits

    old_head = ref['heads']['coarse']
    protected = WholeNetwork(prefix, SuffixReadout(
        healthy, trained, reader, old_head, 'U', True)).eval()
    with torch.no_grad():
        old_logits, protected_logits, current_logits, direct_logits = [], [], [], []
        for indices in order[10000:15000].split(128):
            x = vision.preprocess(images[indices], 224)
            intact = model.encoder(torch.cat([model.class_token.expand(len(x), -1, -1),
                                              model._process_input(x)], dim=1))[:, 0]
            old_logits.append(vision.linear(intact, old_head))
            current, old = protected(x)
            current_logits.append(current)
            protected_logits.append(old)
            direct_logits.append(vision.linear(trained(prefix(x)), old_head))
        old_logits, protected_logits, current_logits, direct_logits = [
            torch.cat(parts) for parts in (old_logits, protected_logits, current_logits, direct_logits)]

    labels = cache['y'][10000:]
    legacy_labels = cache['coarse'][10000:]
    compatibility = evaluate_contracts(protected_logits, old_logits, legacy_labels, ref['thresholds'])
    direct = evaluate_contracts(direct_logits, old_logits, legacy_labels, ref['thresholds'])
    folder = args.out.parent / (args.out.stem + '_states')
    folder.mkdir(exist_ok=True, parents=True)
    output = folder / 'replay.pt'
    torch.save(dict(supervised=baseline_logits, contrastive=logits, native=current_logits,
                    old=old_logits, protected=protected_logits, direct=direct_logits,
                    labels=labels, coarse=legacy_labels, thresholds=ref['thresholds']), output)
    base.save(args.out, dict(complete=True, device='cpu', reference_seed=args.reference_seed,
        seed=6310, checkpoint_sha256=CHECKPOINT_SHA, plan_sha256=args.plan_sha256,
        data_sha256=data_sha, augmentation=augmentation, outcome=outcomes,
        source_sha256={**base.sources(), Path(__file__).name:base.digest(Path(__file__)),
                       Path(contrastive.__file__).name:base.digest(Path(contrastive.__file__))},
        validation_images=len(labels), labeled_images=int(mask.sum()), official_test_loaded=False,
        protected=compatibility, direct=direct, replay_sha256=base.digest(output),
        current_prediction_changes=int((current_logits.argmax(1) != logits.argmax(1)).sum()),
        old_max_logit_error=float((old_logits - protected_logits).abs().max()), branches=[]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
