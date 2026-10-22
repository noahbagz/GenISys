"""Build, train, and sample GenISys using the retained Ada implementations."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHITECTURES = {"ddpm": "monolithic", "hcddpm": "hcddpm", "scddpm": "scddpm"}
RFR_STRENGTHS = {"monolithic": 0.10, "hcddpm": 0.03, "scddpm": 0.10}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare-hull", "build", "train", "sample", "all"))
    parser.add_argument("--architecture", choices=(*ARCHITECTURES, "all"), default="all")
    parser.add_argument("--component", choices=("generator", "classifiers", "rfr", "all"), default="all",
                        help="Training component; classifiers are shared across architectures.")
    parser.add_argument("--settings", type=Path, default=ROOT / "training_utils/paper_settings.json")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "Ship_Systems_Designs_v1p1/Unified_Training_Dataset")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-seed", type=int, default=2001)
    parser.add_argument("--sampling-seed", type=int, default=820000)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--guidance", choices=("none", "constraint", "rfr"), default="rfr")
    parser.add_argument("--range-nmi", type=float, default=5000.0)
    parser.add_argument("--rfr-strength", type=float,
                        help="Default: DDPM/SCDDPM 0.10, HCDDPM 0.03.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run the retained ship evaluator on every sampled attempt.")
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def settings_and_paths(args):
    # Read the existing numerical recipe; only runtime locations, device, and
    # the explicitly requested seed are supplied by this entry point.
    settings = json.loads(args.settings.read_text())
    settings.update(device_name=args.device, training_seed=args.training_seed,
                    training_seeds=[args.training_seed])
    settings["hull_guidance_dataset_root"] = str(args.data_root.resolve() / "Hull_Uniform_50k_v1")
    output = args.output_root.resolve()
    paths = {"unified": args.data_root.resolve(), "output": output,
             "data": output / "data", "generators": output / "generators",
             "classifiers": output / "classifiers" / f"seed_{args.training_seed}" / "feasible_only"}
    for name, path in paths.items():
        if name != "unified":
            path.mkdir(parents=True, exist_ok=True)
    return settings, paths


def prepare_hull(args):
    from training_utils import hull_guidance_data

    # This is the existing 50,000-hull builder and its original random seed.
    # Its module-level input location is relocated to the selected dataset.
    hull_guidance_data.UNIFIED_ROOT = args.data_root.resolve()
    result = hull_guidance_data.build_uniform_hull_data(
        output_root=args.data_root.resolve() / "Hull_Uniform_50k_v1",
    )
    print(json.dumps(result, indent=2, default=str))


def build(data, settings, paths, architectures):
    from models.generators import build_first_experiment_generator
    from training_utils.checkpoints import seed_everything
    from training_utils.model_size import count_trainable_parameters

    summaries = []
    for architecture in architectures:
        seed_everything(settings["training_seed"])
        generator = build_first_experiment_generator(
            data, settings, architecture=architecture, population="feasible_only",
        )
        summaries.append({"architecture": architecture, "denoisers": len(generator.models),
                          "controls": generator.xdim,
                          "trainable_parameters": count_trainable_parameters(generator)})
        del generator
    (paths["output"] / "Model_Build.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


def train(data, settings, paths, architectures, component):
    from models.generators import train_or_load_guidance_classifiers
    from training_utils.training import _train_one_generator

    if component in {"generator", "all"}:
        for architecture in architectures:
            _train_one_generator(data, settings, paths,
                                 population="feasible_only", architecture=architecture)

    if component in {"classifiers", "all"}:
        train_or_load_guidance_classifiers(data, settings, paths["classifiers"],
                                          population="feasible_only")

    if component in {"rfr", "all"}:
        train_rfr(settings, paths, architectures)


def train_rfr(settings, paths, architectures):
    from models.critics import PerformanceCriticConfig, train_or_load_performance_critic
    from training_utils.critic_data import prepare_rfr_quantile_guidance_data
    from training_utils.prefix_critic import train_or_load_prefix_performance_critic
    from training_utils.schema import V1p1ParameterSchema
    from training_utils.subsystems import SUBSYSTEM_ORDER, parameter_subsets

    # Ada's critic builder takes the full-state generator checkpoint's fitted
    # transform. Training only HCDDPM/SCDDPM therefore still needs a previously
    # trained DDPM checkpoint for this particular retained data builder.
    checkpoint = (paths["generators"] / f"seed_{settings['training_seed']}"
                  / "feasible_only/monolithic/checkpoint.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"RFR preparation requires the DDPM coordinate checkpoint: {checkpoint}. "
            "Run train --architecture ddpm --component generator first."
        )
    output = paths["output"] / "rfr"
    critic_data = prepare_rfr_quantile_guidance_data(
        project_root=paths["unified"].parent.parent, output_root=output,
        feasible_generator_checkpoint=checkpoint,
    )
    # These are the original RFR critic training overrides and seeds.
    config = PerformanceCriticConfig(device_name=settings["device_name"],
                                    beta_start=0.001, beta_end=0.2,
                                    training_epochs=40, batches_per_epoch=300,
                                    batch_size=512, training_seed=52001)
    if any(name in architectures for name in ("monolithic", "scddpm")):
        train_or_load_performance_critic(critic_data, config=config,
                                        output_directory=output / "full_state_critic")
    if "hcddpm" in architectures:
        subsets = parameter_subsets(V1p1ParameterSchema().model_names)
        ordered = tuple(subsets[name] for name in SUBSYSTEM_ORDER)
        train_or_load_prefix_performance_critic(
            critic_data, ordered, config=replace(config, training_seed=52002),
            output_directory=output / "prefix_critic_clean_context",
            prefix_context_mode="clean_predecessors",
        )


def load_classifiers(directory, architecture, device):
    from models.classifiers import load_classifier
    from models.generators import GuidanceClassifiers
    from training_utils.subsystems import SUBSYSTEM_ORDER

    # Loading is separate from training: sampling never starts classifier fits.
    local = ({name: load_classifier(directory / f"{architecture}_local/{name}_Classifier.pt", device)
              for name in SUBSYSTEM_ORDER} if architecture != "monolithic" else None)
    return GuidanceClassifiers(
        global_feasibility=load_classifier(directory / "Global_Feasibility_Classifier.pt", device),
        system_level=None,
        hull_feasibility=load_classifier(directory / "Hull_Global_Classifier.pt", device)
                        if architecture == "monolithic" else None,
        hcddpm_local=local if architecture == "hcddpm" else None,
        scddpm_local=local if architecture == "scddpm" else None,
    )


def sample(data, settings, paths, architectures, args):
    import numpy as np
    import pandas as pd
    import torch

    from models.generators import GuidanceClassifiers
    from models.noise import NoiseStream
    from models.sampling import SAMPLERS
    from models.critics import load_performance_critic_checkpoint
    from training_utils.prefix_critic import load_prefix_performance_critic_checkpoint
    from training_utils.schema import V1p1ParameterSchema
    from training_utils.training import _load_generator

    schema = V1p1ParameterSchema()
    range_index = schema.model_names.index("Voyage_Range")
    reference = data.model_values[data.generator_positions("feasible_only")[0]]
    controls = np.repeat(reference.reshape(1, -1), args.count, axis=0)
    controls[:, range_index] = args.range_nmi
    editable = np.delete(np.arange(133), range_index)

    for architecture in architectures:
        directory = (paths["output"] / "samples" / architecture / args.guidance
                     / f"range_{args.range_nmi:g}" / f"seed_{args.sampling_seed}")
        generator = _load_generator(data, settings, paths,
                                    population="feasible_only", architecture=architecture)
        for environment in generator.environments:
            # Retain the publication fixed-range sampler's deterministic t=0.
            environment.DDPM_Dict["sampling_omit_t0"] = False

        if args.guidance == "none":
            # The unchanged masked samplers call eval()/local_for() even when
            # guidance is disabled. No-op modules satisfy that interface;
            # their forward functions are never called at alpha=gamma=0.
            neutral = torch.nn.Identity()
            from training_utils.subsystems import SUBSYSTEM_ORDER
            local = dict.fromkeys(SUBSYSTEM_ORDER, neutral)
            classifiers = GuidanceClassifiers(neutral, None, hull_feasibility=neutral,
                                              hcddpm_local=local, scddpm_local=local)
        else:
            classifiers = load_classifiers(paths["classifiers"], architecture, args.device)

        critic = torch.nn.Identity()
        gamma = 0.0
        if args.guidance == "rfr":
            gamma = args.rfr_strength if args.rfr_strength is not None else RFR_STRENGTHS[architecture]
            if architecture == "hcddpm":
                loader = load_prefix_performance_critic_checkpoint
                checkpoint = paths["output"] / "rfr/prefix_critic_clean_context/best_prefix_performance_critic.pt"
            else:
                loader = load_performance_critic_checkpoint
                checkpoint = paths["output"] / "rfr/full_state_critic/best_performance_critic.pt"
            critic, _ = loader(checkpoint, device_name=args.device)

        rng = NoiseStream(args.sampling_seed, generator.device)
        # Preserve prior attempts after all required assets have loaded.
        directory.mkdir(parents=True, exist_ok=False)
        generated, report = SAMPLERS[architecture](
            generator, classifiers, critic, controls, editable,
            feasibility_alpha=0.0 if args.guidance == "none" else 0.7,
            performance_alpha=gamma, objective_index=0, rng=rng,
        )
        decoded, changes = schema.decode(generated)
        pd.DataFrame(generated, columns=schema.model_names).to_csv(directory / "Controls_133.csv", index=False)
        pd.DataFrame(decoded, columns=schema.full_names).to_csv(directory / "Designs_189.csv", index=False)
        changes.to_csv(directory / "Postprocess_Changes.csv", index=False)
        sampler_report = asdict(report)
        if not np.isfinite(sampler_report["maximum_absolute_latent"]):
            sampler_report["maximum_absolute_latent"] = None
        receipt = {"architecture": architecture, "guidance": args.guidance,
                   "range_nmi": args.range_nmi, "sampling_seed": args.sampling_seed,
                   "training_seed": args.training_seed, "rfr_strength": gamma,
                   "requested": args.count, "sampler": sampler_report, "rng": rng.report()}
        (directory / "Sampling_Report.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(directory)

        if args.evaluate:
            from eval_utils.evaluate_samples import evaluate_generated_samples
            evaluate_generated_samples(decoded, directory / "evaluation", workers=args.workers)
        del generator, classifiers, critic


def main():
    args = arguments()
    if args.stage == "prepare-hull":
        prepare_hull(args)
        return

    from training_utils.training import _prepare_data

    settings, paths = settings_and_paths(args)
    data = _prepare_data(ROOT, settings, paths)
    architectures = list(ARCHITECTURES.values()) if args.architecture == "all" else [ARCHITECTURES[args.architecture]]
    if args.stage == "build":
        build(data, settings, paths, architectures)
    elif args.stage in {"train", "all"}:
        train(data, settings, paths, architectures, args.component)
    if args.stage in {"sample", "all"}:
        sample(data, settings, paths, architectures, args)


if __name__ == "__main__":
    main()
