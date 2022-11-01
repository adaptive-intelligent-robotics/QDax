# QDax Caveats

Here is a few caveats one should be aware of when using QDax.

## Use of auto reset for Brax environments
The use of `auto_reset` can be tricky and lead to problems and/or unwanted behaviors. By defaults in our examples, we set auto reset equals True, so the samples collected in the Replay Buffer are good quality samples and stay within the distribution of interest. This is particularly important in the case of PGAME, where putting auto reset to False could lead to important decrease in data efficiency and final performance.

## In-place replacement of state descriptors in QDTransition
The state descriptor from Brax environments is stored in a dictionary. The retrievement of this data when building the QDTransition in a step is hence tricky. The state descriptor must be stored in a variable before applying a environment step, because an in-place replacement is going to occur during the step function of Brax.

One should take inspiration from the `play_step` function from our examples.
