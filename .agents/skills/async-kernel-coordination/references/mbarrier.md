# mbarrier: Async Coordination

An `mbarrier` (memory barrier) is a hardware synchronization object stored in
shared memory. Its internal encoding is opaque; to use it you track three
fields:

- **Arrival count** — arrivals still missing in the current round.
- **Phase** — the current round. You track only **phase parity** (0 or 1).
- **tx-count** — bytes still in flight, used when TMA is involved.

## init

`mbarrier.init.shared::cta.b64 [ptr], expected_arrivals` sets the expected
arrival count; the barrier starts in **phase 0** with the pending count equal
to `expected_arrivals`. It is now waiting for the relevant producers or
resource users to report completion.

## arrive

Every arrival reduces the count of work the barrier is still waiting for.
Different participants arrive differently:

- **`mbarrier.arrive.expect_tx(bytes)`** — the TMA path. It does two things:
  (1) the issuing thread contributes **one arrival**, and (2) it adds `bytes`
  to the tx-count. The thread's arrival does **not** mean the barrier is
  complete: as the TMA engine finishes each transfer it applies
  **complete-tx** updates that subtract from the tx-count. The phase completes
  only when **arrival count == 0 AND tx-count == 0**, then phase parity flips.
- **`tcgen05.commit...mbarrier::arrive`** — the Tensor-Core path. Issuing
  `tcgen05.mma` alone does **not** update a barrier; this commit associates one
  arrival with the previously issued async tcgen05 operations, and hardware
  reports it only after they complete. Without the commit, a waiting consumer
  cannot make progress.
- **`mbarrier.arrive`** — a plain thread arrival. A consumer that has finished
  reading a buffer can arrive to tell the producer the buffer may be
  overwritten and reused.

## wait

`wait` is the consumer side of the same protocol. The consumer waits the phase
associated with the current iteration and may read data or reuse a resource
only after it completes. Raw PTX `mbarrier.try_wait.parity` may return `false`
**before** the phase completes and must be retried; wrap it in a loop that
blocks until the requested phase completes.

Because `arrive` and `wait` are separate, a producer can report progress and
continue with other work, while the consumer waits only when it actually needs
the result.

## Phases distinguish consecutive uses

The same mbarrier is reused across rounds. After one phase completes the
barrier enters the next and begins waiting for a new set of arrivals. If a
consumer recorded only that the barrier had completed *at some point*, it
could mistake the previous round's completion for the current data being
ready. **Phase parity** alternates 0 ↔ 1 so the consumer identifies the
particular completion it is waiting for.

A two-stage pipeline has one barrier per stage and tracks a `phase_tma` that
flips only after both stages are visited:

```
stage = iteration % 2
try_wait(tma_bar[stage], phase_tma)
if stage == 1: phase_tma ^= 1
```

| iteration | stage | parity waited | parity after | phase_tma after |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 0 |
| 1 | 1 | 0 | 1 | 1 |
| 2 | 0 | 1 | 0 | 1 |
| 3 | 1 | 1 | 0 | 0 |

`phase_tma` describes software progress through the circular buffer; it does
not assume the two transfers complete in any particular hardware order.

## full and empty barriers

A barrier can mean either "data is ready" or "the buffer is no longer in use."
A pipelined kernel therefore gives **each** SMEM stage a pair:

- `full[stage]` — TMA filled it; producer → consumer.
- `empty[stage]` — consumer finished; consumer → producer.

Once the pipeline reaches steady state, a stage cycles:

```
producer: wait empty[stage] → fill → arrive full[stage]
consumer: wait full[stage]  → read → arrive empty[stage]
```

Track the `full` and `empty` phase parities **separately**. Each barrier's
expected arrival count depends on how many threads report completion.

## The three common handoffs

When reading waits and arrivals in a kernel, first identify the **producer**,
the **consumer**, and the **data or resource** being handed off.

1. **Threads → async hardware.** If threads write SMEM and a later TMA store
   or MMA reads it, establish the required synchronization and ordering first;
   otherwise the async operation may begin reading before the writes finish.
2. **TMA → MMA.** The producer (one thread) registers the transfer bytes via
   `expect_tx`; the MMA consumer waits the current barrier phase, plus any
   ordering the instruction requires, before reading the tile.
3. **MMA → epilogue.** `tcgen05.commit` on an mbarrier signals the accumulator
   is complete; the epilogue waits that barrier and applies the required
   **tcgen05 fence** before reading TMEM.

## Proxy fences

CLC writes its response through the **async proxy** while an ordinary thread
reads it through the **generic proxy**. The mbarrier wait confirms the async
write completed, but PTX additionally requires `fence.proxy.async` fences
(before submitting a new request and after reading the response) to order
accesses to the response buffer across the two proxies, preventing a later
async write from racing data still being read. Real code must also manage the
barrier phase and the required CTA- or cluster-wide thread synchronization.
