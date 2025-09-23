#include "SparseNondeterministicInfiniteHorizonHelper.h"

#include "storm/modelchecker/helper/infinitehorizon/internal/ComponentUtility.h"
#include "storm/modelchecker/helper/infinitehorizon/internal/LraViHelper.h"
#include "storm/modelchecker/helper/infinitehorizon/SparseDeterministicInfiniteHorizonHelper.h"
#include "storm/utility/matrix.h"
#include "storm/storage/SparseMatrix.h"

#include "storm/storage/MaximalEndComponentDecomposition.h"
#include "storm/storage/Scheduler.h"
#include "storm/storage/SchedulerChoice.h"
#include "storm/storage/SparseMatrix.h"

#include "storm/solver/LpSolver.h"
#include "storm/solver/MinMaxLinearEquationSolver.h"
#include "storm/solver/multiplier/Multiplier.h"

#include "storm/utility/solver.h"
#include "storm/utility/vector.h"

#include "storm/environment/solver/LongRunAverageSolverEnvironment.h"
#include "storm/environment/solver/MinMaxSolverEnvironment.h"

#include <boost/spirit/home/qi/action/action.hpp>
#include <iostream>  // For debugging
#include "storm/exceptions/UnmetRequirementException.h"
#include <filesystem>

#include "spot/twa/twa.hh"
using namespace std;

namespace storm {
namespace modelchecker {
namespace helper {

uint64_t componentNumber = 0; // Used for naming files during export of end component certificates.

template<typename ValueType>
SparseNondeterministicInfiniteHorizonHelper<ValueType>::SparseNondeterministicInfiniteHorizonHelper(
    storm::storage::SparseMatrix<ValueType> const& transitionMatrix)
    : SparseInfiniteHorizonHelper<ValueType, true>(transitionMatrix) {
    // Intentionally left empty.
}

template<typename ValueType>
SparseNondeterministicInfiniteHorizonHelper<ValueType>::SparseNondeterministicInfiniteHorizonHelper(
    storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector const& markovianStates, std::vector<ValueType> const& exitRates)
    : SparseInfiniteHorizonHelper<ValueType, true>(transitionMatrix, markovianStates, exitRates) {
    // Intentionally left empty.
}

template<typename ValueType>
std::vector<uint64_t> const& SparseNondeterministicInfiniteHorizonHelper<ValueType>::getProducedOptimalChoices() const {
    STORM_LOG_ASSERT(this->isProduceSchedulerSet(), "Trying to get the produced optimal choices although no scheduler was requested.");
    STORM_LOG_ASSERT(this->_producedOptimalChoices.is_initialized(),
                     "Trying to get the produced optimal choices but none were available. Was there a computation call before?");
    return this->_producedOptimalChoices.get();
}

template<typename ValueType>
std::vector<uint64_t>& SparseNondeterministicInfiniteHorizonHelper<ValueType>::getProducedOptimalChoices() {
    STORM_LOG_ASSERT(this->isProduceSchedulerSet(), "Trying to get the produced optimal choices although no scheduler was requested.");
    STORM_LOG_ASSERT(this->_producedOptimalChoices.is_initialized(),
                     "Trying to get the produced optimal choices but none were available. Was there a computation call before?");
    return this->_producedOptimalChoices.get();
}

template<typename ValueType>
storm::storage::Scheduler<ValueType> SparseNondeterministicInfiniteHorizonHelper<ValueType>::extractScheduler() const {
    auto const& optimalChoices = getProducedOptimalChoices();
    storm::storage::Scheduler<ValueType> scheduler(optimalChoices.size());
    for (uint64_t state = 0; state < optimalChoices.size(); ++state) {
        scheduler.setChoice(optimalChoices[state], state);
    }
    return scheduler;
}

template<typename ValueType>
void SparseNondeterministicInfiniteHorizonHelper<ValueType>::createDecomposition() {
    if (this->_longRunComponentDecomposition == nullptr) {
        // The decomposition has not been provided or computed, yet.
        this->createBackwardTransitions();
        this->_computedLongRunComponentDecomposition =
            std::make_unique<storm::storage::MaximalEndComponentDecomposition<ValueType>>(this->_transitionMatrix, *this->_backwardTransitions);
        this->_longRunComponentDecomposition = this->_computedLongRunComponentDecomposition.get();
    }
}

template<typename ValueType>
ValueType SparseNondeterministicInfiniteHorizonHelper<ValueType>::computeLraForComponent(Environment const& env, ValueGetter const& stateRewardsGetter,
                                                                                         ValueGetter const& actionRewardsGetter,
                                                                                         storm::storage::MaximalEndComponent const& component) {
    // For models with potential nondeterminisim, we compute the LRA for a maximal end component (MEC)

    // Allocate memory for the nondeterministic choices.
    if (this->isProduceSchedulerSet()) {
        if (!this->_producedOptimalChoices.is_initialized()) {
            this->_producedOptimalChoices.emplace();
        }
        this->_producedOptimalChoices->resize(this->_transitionMatrix.getRowGroupCount());
    }

    auto trivialResult = this->computeLraForTrivialMec(env, stateRewardsGetter, actionRewardsGetter, component);
    if (trivialResult.first) {
        return trivialResult.second;
    }

    // Solve nontrivial MEC with the method specified in the settings
    storm::solver::LraMethod method = env.solver().lra().getNondetLraMethod();
    if ((storm::NumberTraits<ValueType>::IsExact || env.solver().isForceExact()) && env.solver().lra().isNondetLraMethodSetFromDefault() &&
        method != storm::solver::LraMethod::LinearProgramming) {
        STORM_LOG_INFO(
            "Selecting 'LP' as the solution technique for long-run properties to guarantee exact results. If you want to override this, please explicitly "
            "specify a different LRA method.");
        method = storm::solver::LraMethod::LinearProgramming;
        } else if (env.solver().isForceSoundness() && env.solver().lra().isNondetLraMethodSetFromDefault() && method != storm::solver::LraMethod::ValueIteration) {
            STORM_LOG_INFO(
                "Selecting 'VI' as the solution technique for long-run properties to guarantee sound results. If you want to override this, please explicitly "
                "specify a different LRA method.");
            method = storm::solver::LraMethod::ValueIteration;
        }
    // STORM_LOG_ERROR_COND(!this->isProduceSchedulerSet() || method == storm::solver::LraMethod::ValueIteration,
    //                      "Scheduler generation not supported for the chosen LRA method. Try value-iteration.");
    if (method == storm::solver::LraMethod::LinearProgramming) {
        return computeLraForMecLp(env, stateRewardsGetter, actionRewardsGetter, component);
    } else if (method == storm::solver::LraMethod::ValueIteration) {
        return computeLraForMecVi(env, stateRewardsGetter, actionRewardsGetter, component);
    } else if (method == storm::solver::LraMethod::PolicyIteration) {
        vector<ValueType> biases; // For certificate export.
        storm::storage::SparseMatrix<ValueType> mecMatrix;
        ValueType gain = computeLraForMecPi(env, stateRewardsGetter, actionRewardsGetter, component, biases, mecMatrix);
        // if (env.solver().lra().getCertificateFilename() != "") { // TODO actually use this check and make it work with CLI
        //
        // }
        // Check the certificate in Storm.
        namespace fs = std::filesystem;
        fs::create_directories("./certificates");

        // mecStatesToGlobalStates
        auto stateSet = component.getStateSet(); // Need to store it in a variable first, otherwise it breaks...
        auto mecStatesToGlobalStates = std::vector<uint64_t> (stateSet.begin(), stateSet.end());
        auto mecRowsToGlobalRows = std::vector<uint64_t>();
        for (const auto& state : component.getStateSet()) {
            for (const auto& choice : component.getChoicesForState(state)) {
                mecRowsToGlobalRows.push_back(choice);
            }
        }
        ranges::sort(mecStatesToGlobalStates);
        ranges::sort(mecRowsToGlobalRows);


        ValueType certificateInterval = 1e-5; // TODO extract from settings.
        bool isValid = checkMecCertificate(component, gain - certificateInterval, gain + certificateInterval, biases, stateRewardsGetter, actionRewardsGetter);
        if (isValid) { cout << "Certificate OK" << endl; } else { cout << "Certificate wrong!" << endl; abort(); }
        ofstream endComponentFile;
        endComponentFile.open("certificates/" + std::to_string(componentNumber) + ".mec");
        mecMatrix.printAsMatlabMatrix(endComponentFile);
        endComponentFile.close();

        ofstream stateRewardsFile;
        stateRewardsFile.open("certificates/" + std::to_string(componentNumber) + ".srew");
        for (auto const& state : mecStatesToGlobalStates) {
            stateRewardsFile << stateRewardsGetter(state) << endl;
        }
        stateRewardsFile.close();

        ofstream actionRewardsFile;
        actionRewardsFile.open("certificates/" + std::to_string(componentNumber) + ".arew");
        for (auto const& choice : mecRowsToGlobalRows) {
            actionRewardsFile << actionRewardsGetter(choice) << endl;
        }

        actionRewardsFile.close();

        ofstream certificateFile;
        certificateFile.open("certificates/" + std::to_string(componentNumber) + ".cert");
        certificateFile << this->getOptimizationDirection() << endl;
        certificateFile << gain << endl;
        for (auto const& b : biases) {
            certificateFile << b << std::endl;
        }
        certificateFile.close();
        componentNumber++;
        return gain;
    } else {
        STORM_LOG_THROW(false, storm::exceptions::InvalidSettingsException, "Unsupported technique.");
    }
}

template<typename ValueType>
std::pair<bool, ValueType> SparseNondeterministicInfiniteHorizonHelper<ValueType>::computeLraForTrivialMec(
    Environment const& env, ValueGetter const& stateRewardsGetter, ValueGetter const& actionRewardsGetter,
    storm::storage::MaximalEndComponent const& component) {
    // If the component only consists of a single state, we compute the LRA value directly
    if (component.size() == 1) {
        auto const& element = *component.begin();
        uint64_t state = internal::getComponentElementState(element);
        auto choiceIt = internal::getComponentElementChoicesBegin(element);
        if (!this->isContinuousTime()) {
            // This is an MDP.
            // Find the choice with the highest/lowest reward
            ValueType bestValue = actionRewardsGetter(*choiceIt);
            uint64_t bestChoice = *choiceIt;
            for (++choiceIt; choiceIt != internal::getComponentElementChoicesEnd(element); ++choiceIt) {
                ValueType currentValue = actionRewardsGetter(*choiceIt);
                if ((this->minimize() && currentValue < bestValue) || (this->maximize() && currentValue > bestValue)) {
                    bestValue = std::move(currentValue);
                    bestChoice = *choiceIt;
                }
            }
            if (this->isProduceSchedulerSet()) {
                this->_producedOptimalChoices.get()[state] = bestChoice - this->_transitionMatrix.getRowGroupIndices()[state];
            }
            bestValue += stateRewardsGetter(state);
            return {true, bestValue};
        } else {
            // In a Markov Automaton, singleton components have to consist of a Markovian state because of the non-Zenoness assumption. Then, there is just one
            // possible choice.
            STORM_LOG_ASSERT(this->_markovianStates != nullptr,
                             "Nondeterministic continuous time model without Markovian states... Is this a not a Markov Automaton?");
            STORM_LOG_THROW(this->_markovianStates->get(state), storm::exceptions::InvalidOperationException,
                            "Markov Automaton has Zeno behavior. Computation of Long Run Average values not supported.");
            STORM_LOG_ASSERT(internal::getComponentElementChoiceCount(element) == 1, "Markovian state has Nondeterministic behavior.");
            if (this->isProduceSchedulerSet()) {
                this->_producedOptimalChoices.get()[state] = 0;
            }
            ValueType result = stateRewardsGetter(state) +
                               (this->isContinuousTime() ? (*this->_exitRates)[state] * actionRewardsGetter(*choiceIt) : actionRewardsGetter(*choiceIt));
            return {true, result};
        }
    }
    return {false, storm::utility::zero<ValueType>()};
}

template<typename ValueType>
ValueType SparseNondeterministicInfiniteHorizonHelper<ValueType>::computeLraForMecVi(Environment const& env, ValueGetter const& stateRewardsGetter,
                                                                                     ValueGetter const& actionRewardsGetter,
                                                                                     storm::storage::MaximalEndComponent const& mec) {
    // Collect some parameters of the computation
    ValueType aperiodicFactor = storm::utility::convertNumber<ValueType>(env.solver().lra().getAperiodicFactor());
    std::vector<uint64_t>* optimalChoices = nullptr;
    if (this->isProduceSchedulerSet()) {
        optimalChoices = &this->_producedOptimalChoices.get();
    }

    // Now create a helper and perform the algorithm
    if (this->isContinuousTime()) {
        // We assume a Markov Automaton (with deterministic timed states and nondeterministic instant states)
        storm::modelchecker::helper::internal::LraViHelper<ValueType, storm::storage::MaximalEndComponent,
                                                           storm::modelchecker::helper::internal::LraViTransitionsType::DetTsNondetIs>
            viHelper(mec, this->_transitionMatrix, aperiodicFactor, this->_markovianStates, this->_exitRates);
        return viHelper.performValueIteration(env, stateRewardsGetter, actionRewardsGetter, this->_exitRates, &this->getOptimizationDirection(),
                                              optimalChoices);
    } else {
        // We assume an MDP (with nondeterministic timed states and no instant states)
        storm::modelchecker::helper::internal::LraViHelper<ValueType, storm::storage::MaximalEndComponent,
                                                           storm::modelchecker::helper::internal::LraViTransitionsType::NondetTsNoIs>
            viHelper(mec, this->_transitionMatrix, aperiodicFactor);
        return viHelper.performValueIteration(env, stateRewardsGetter, actionRewardsGetter, nullptr, &this->getOptimizationDirection(), optimalChoices);
    }
}

template<typename ValueType>
ValueType SparseNondeterministicInfiniteHorizonHelper<ValueType>::computeLraForMecLp(Environment const& env, ValueGetter const& stateRewardsGetter,
                                                                                     ValueGetter const& actionRewardsGetter,
                                                                                     storm::storage::MaximalEndComponent const& mec) {
    // Create an LP solver
    auto solver = storm::utility::solver::getLpSolver<ValueType>("LRA for MEC");

    // Now build the LP formulation as described in:
    // Guck et al.: Modelling and Analysis of Markov Reward Automata (ATVA'14), https://doi.org/10.1007/978-3-319-11936-6_13
    solver->setOptimizationDirection(invert(this->getOptimizationDirection()));

    // Create variables
    // TODO: Investigate whether we can easily make the variables bounded
    std::map<uint_fast64_t, storm::expressions::Variable> stateToVariableMap;
    for (auto const& stateChoicesPair : mec) {
        std::string variableName = "x" + std::to_string(stateChoicesPair.first);
        stateToVariableMap[stateChoicesPair.first] = solver->addUnboundedContinuousVariable(variableName);
    }
    storm::expressions::Variable k = solver->addUnboundedContinuousVariable("k", storm::utility::one<ValueType>());
    solver->update();

    // Add constraints.
    for (auto const& stateChoicesPair : mec) {
        uint_fast64_t state = stateChoicesPair.first;
        bool stateIsMarkovian = this->_markovianStates && this->_markovianStates->get(state);

        // Now create a suitable constraint for each choice
        // x_s  {≤, ≥}  -k/rate(s) + sum_s' P(s,act,s') * x_s' + (value(s)/rate(s) + value(s,act))
        for (auto choice : stateChoicesPair.second) {
            std::vector<storm::expressions::Expression> summands;
            auto matrixRow = this->_transitionMatrix.getRow(choice);
            summands.reserve(matrixRow.getNumberOfEntries() + 2);
            // add -k/rate(s) (only if s is either a Markovian state or we have an MDP)
            if (stateIsMarkovian) {
                summands.push_back(-(k / solver->getManager().rational((*this->_exitRates)[state])));
            } else if (!this->isContinuousTime()) {
                summands.push_back(-k);
            }
            // add sum_s' P(s,act,s') * x_s'
            for (auto const& element : matrixRow) {
                summands.push_back(stateToVariableMap.at(element.getColumn()) * solver->getConstant(element.getValue()));
            }
            // add value for state and choice
            ValueType value;
            if (stateIsMarkovian) {
                // divide state reward with exit rate
                value = stateRewardsGetter(state) / (*this->_exitRates)[state] + actionRewardsGetter(choice);
            } else if (!this->isContinuousTime()) {
                // in discrete time models no scaling is needed
                value = stateRewardsGetter(state) + actionRewardsGetter(choice);
            } else {
                // state is a probabilistic state of a Markov automaton. The state reward is not collected
                value = actionRewardsGetter(choice);
            }
            summands.push_back(solver->getConstant(value));
            storm::expressions::Expression constraint;
            if (this->minimize()) {
                constraint = stateToVariableMap.at(state) <= storm::expressions::sum(summands);
            } else {
                constraint = stateToVariableMap.at(state) >= storm::expressions::sum(summands);
            }
            solver->addConstraint("s" + std::to_string(state) + "," + std::to_string(choice), constraint);
        }
    }

    solver->optimize();
    STORM_LOG_THROW(!this->isProduceSchedulerSet(), storm::exceptions::NotImplementedException,
                    "Scheduler extraction is not yet implemented for LP based LRA method.");
    return solver->getContinuousValue(k);
}

// Like capital E from the paper!
template<typename ValueType>
ValueType computeWeightedSuccSum(storm::storage::SparseMatrix<ValueType> mecMatrix, uint64_t row, std::vector<ValueType> gains) {
    auto sum = storm::utility::zero<ValueType>();
    for (const auto& succ : mecMatrix.getRow(row)) {
        ValueType p = succ.getValue();
        auto succCol = succ.getColumn();
        sum = sum + p * gains[succCol];;
    }
    return sum;
}

template<typename ValueType>
pair<bool, std::vector<uint64_t>> gainImprovementStep(const storm::storage::SparseMatrix<ValueType>& mecMatrix, std::vector<uint64_t> scheduler,
        const std::vector<ValueType>& gains) {
    // We consider all possible choices and see if we can greedily improve the gain for a state, basically an implementation of arg max.
    bool schedulerWasChanged = false;
    for (uint64_t state = 0; state < mecMatrix.getRowGroupCount(); state++) {
        auto stateRow = mecMatrix.getRowGroupIndices()[state];
        auto numChoices = mecMatrix.getRowGroupSize(state);

        auto previousRow = scheduler[state];
        auto bestRow = scheduler[state]; // If nothing better than this choice is found, we keep it like it is.
        ValueType bestScore = computeWeightedSuccSum(mecMatrix, bestRow, gains);

        for (auto offset = 0; offset < numChoices; offset++) {
            auto currentRow = stateRow + offset;
            ValueType currentScore = computeWeightedSuccSum(mecMatrix, currentRow, gains);
            // Save the best choice and score, with preference for the previous choice if possible.
            if (currentScore > bestScore) {
                bestRow = currentRow;
                bestScore = currentScore;
            }
        }
        if (bestRow != previousRow) {
            schedulerWasChanged = true;
            scheduler[state] = bestRow;
            cout << state << " g-> " << bestRow << endl;
        }
    }
    return std::pair(schedulerWasChanged, scheduler);
}

template<typename ValueType>
pair<bool, std::vector<uint64_t>> biasImprovementStep(const storm::storage::SparseMatrix<ValueType>& mecMatrix, std::vector<uint64_t> scheduler,
        const std::vector<ValueType>& biases, const std::function<ValueType(uint64_t)>& mecStateRewardsGetter, std::function<ValueType(uint64_t)>& mecActionRewardsGetter) {
    // We consider all possible choices and see if we can greedily improve the gain for a state, basically an implementation of arg max.
    bool schedulerWasChanged = false;
    for (uint64_t state = 0; state < mecMatrix.getRowGroupCount(); state++) {
        auto stateRow = mecMatrix.getRowGroupIndices()[state];
        auto numChoices = mecMatrix.getRowGroupSize(state);

        auto previousRow = scheduler[state];
        auto bestRow = scheduler[state]; // If nothing better than this choice is found, we keep it like it is.
        ValueType bestScore = computeWeightedSuccSum(mecMatrix, bestRow, biases) + mecStateRewardsGetter(state) + mecActionRewardsGetter(bestRow);

        for (auto offset = 0; offset < numChoices; offset++) {
            auto currentRow = stateRow + offset;
            ValueType currentScore = computeWeightedSuccSum(mecMatrix, currentRow, biases) + mecStateRewardsGetter(state) + mecActionRewardsGetter(currentRow);
            // Save the best choice and score, with preference for the previous choice if possible.
            if (currentScore > bestScore) {
                bestRow = currentRow;
                bestScore = currentScore;
            }
        }
        if (bestRow != previousRow) {
            schedulerWasChanged = true;
            scheduler[state] = bestRow;
            cout << state << " b-> " << bestRow << endl;
        }
    }
    return std::pair(schedulerWasChanged, scheduler);
}

// Apply the scheduler to the matrix, and return the modified actionrewardsgetter.
template<typename ValueType>
std::pair<storm::storage::SparseMatrix<ValueType>, std::function<ValueType(uint64_t)>> applyScheduler(
        storm::storage::SparseMatrix<ValueType>& mecMatrix, std::vector<uint64_t>& scheduler, std::function<ValueType(uint64_t)> const& mecActionRewardsGetter) {
    storm::storage::BitVector schedulerBitVector(mecMatrix.getRowCount());
    for (const auto& row : scheduler) { schedulerBitVector.set(row); }
    auto detMatrix = mecMatrix.restrictRows(schedulerBitVector);
    std::function<ValueType(uint64_t)> detActionRewardsGetter = [&scheduler, &mecActionRewardsGetter](uint64_t detState) -> ValueType {
        // Scheduler is a map from mec states to the chosen rows
        // The mec states and det states are the same (in the detMatrix row = state)
        // So it's also a map from the det states to the chosen rows, which makes this correct.
        return mecActionRewardsGetter(scheduler[detState]);
    };
    return std::pair(detMatrix, detActionRewardsGetter);
}

// TODO reference diane's paper here
template<typename ValueType>
bool SparseNondeterministicInfiniteHorizonHelper<ValueType>::checkMecCertificate(storm::storage::MaximalEndComponent const& mec,
        ValueType const lowerBound, ValueType const upperBound, std::vector<ValueType> const& certificateVector,
        ValueGetter const& stateRewardsGetter, ValueGetter const& actionRewardsGetter) {
    auto stateSet = mec.getStateSet(); // Need to store it in a variable first, otherwise it breaks...
    auto mecStatesToGlobalStates = std::vector<uint64_t> (stateSet.begin(), stateSet.end());
    ranges::sort(mecStatesToGlobalStates);
    auto globalStatesToMecStates = std::map<uint64_t, uint64_t>();
    for (uint64_t mecState = 0; mecState < stateSet.size(); mecState++) {
        globalStatesToMecStates[mecStatesToGlobalStates[mecState]] = mecState;
    }

    ValueType optimizationDirectionFactor = this->getOptimizationDirection() == storm::solver::OptimizationDirection::Maximize ? storm::utility::one<ValueType>() : -storm::utility::one<ValueType>();
    if (optimizationDirectionFactor < 0) {
        cout << endl;
    }

    bool isValid = true;
    for (uint64_t mecStateIndex = 0; mecStateIndex < stateSet.size(); mecStateIndex++ ) {
        uint64_t globalState = mecStatesToGlobalStates[mecStateIndex];
        ValueType bestScore = - storm::utility::infinity<ValueType>();
        for (const uint64_t& choice: mec.getChoicesForState(globalState)) {
            ValueType currentScore = storm::utility::zero<ValueType>();
            for (const auto& successor : this->_transitionMatrix.getRow(choice)) {
                currentScore += optimizationDirectionFactor * successor.getValue() * certificateVector[globalStatesToMecStates[successor.getColumn()]];
            }
            currentScore += optimizationDirectionFactor * (stateRewardsGetter(globalState) + actionRewardsGetter(choice));
            if (currentScore > bestScore) { bestScore = currentScore; }
        }
        if (mecStateIndex == -1 && componentNumber == -1) {
            cout << "operator result: " << bestScore << endl;
            cout << "cv: " << certificateVector[mecStateIndex] << endl;
            abort();
        }
        bestScore -= optimizationDirectionFactor * certificateVector[mecStateIndex];
        isValid &= lowerBound <= optimizationDirectionFactor * bestScore && optimizationDirectionFactor * bestScore <= upperBound;
    }
    return isValid;
}

template<typename ValueType>
ValueType SparseNondeterministicInfiniteHorizonHelper<ValueType>::computeLraForMecPi(Environment const& env, ValueGetter const& stateRewardsGetter,
                                                                                     ValueGetter const& actionRewardsGetter,
                                                                                     storm::storage::MaximalEndComponent const& mec,
                                                                                     std::vector<ValueType>& biases,
                                                                                     storm::storage::SparseMatrix<ValueType>& mecMatrix) {
    // Note that minimizing a reward is handled by making the rewards negative and maximizing over that.

    // mecStatesToGlobalStates
    auto stateSet = mec.getStateSet(); // Need to store it in a variable first, otherwise it breaks...
    auto mecStatesToGlobalStates = std::vector<uint64_t> (stateSet.begin(), stateSet.end());
    auto mecRowsToGlobalRows = std::vector<uint64_t>();

    // First of all, we create a matrix for just the MEC. We limit it to only the choices that the MEC allows.
    auto mecStateBitVector = storm::storage::BitVector(this->_transitionMatrix.getRowGroupCount());
    auto mecActionBitVector = storm::storage::BitVector(this->_transitionMatrix.getRowCount());
    for (const auto& state : mec.getStateSet()) {
        mecStateBitVector.set(state);
        for (const auto& choice : mec.getChoicesForState(state)) {
            mecRowsToGlobalRows.push_back(choice);
            mecActionBitVector.set(choice);
        }
    }
    ranges::sort(mecStatesToGlobalStates);
    ranges::sort(mecRowsToGlobalRows);

    mecMatrix = this->_transitionMatrix.getSubmatrix(false,  mecActionBitVector, mecStateBitVector);
    /*{
        cout << "MEC:" << endl;
        for (const auto& state : mec.getStateSet()) {
            cout << state << ": ";
            for (const auto& choice : mec.getChoicesForState(state)) { cout << choice << " "; }
            cout << endl;
        }
        cout << "original" << endl;
        //this->_transitionMatrix.printAsMatlabMatrix(cout);
        cout << "mecMatrix" << endl;
        mecMatrix.printAsMatlabMatrix(cout);
        for (uint64_t rowIndex = 0; rowIndex < mecMatrix.getRowCount(); rowIndex++) {
             if (mecMatrix.getRow(rowIndex).getNumberOfEntries() == 0) {
                 cout << "Zero row in MEC, should never happen." << endl;
                 abort();
             }
        }
    }*/ // Debug: check if making the MEC Matrix was a success.

    ValueType optimizationDirectionFactor = this->getOptimizationDirection() == storm::solver::OptimizationDirection::Maximize ? storm::utility::one<ValueType>() : -storm::utility::one<ValueType>();
    std::function<ValueType(uint64_t)> mecStateRewardsGetter = [&optimizationDirectionFactor, &stateRewardsGetter, mecStatesToGlobalStates] (const uint64_t mecState) {
        ValueType res = optimizationDirectionFactor * stateRewardsGetter(mecStatesToGlobalStates[mecState]); // Do NOT simplify, it will crash sometimes.
        return res;
    };

    std::function<ValueType(uint64_t)> mecActionRewardsGetter = [&optimizationDirectionFactor, &actionRewardsGetter, &mecRowsToGlobalRows] (const uint64_t mecRow) {
        ValueType res = optimizationDirectionFactor * actionRewardsGetter(mecRowsToGlobalRows[mecRow]); // Do NOT simplify, it will crash sometimes.
        return res;
    };

    /*{
        cout << "MEC state, global state, rewards should match" << endl;
        for (uint64_t i = 0; i < mecStatesToGlobalStates.size(); i++) {
            cout << i << " " << mecStatesToGlobalStates[i] << " " << stateRewardsGetter(mecStatesToGlobalStates[i]) << " " << mecStateRewardsGetter(i) << endl;
            if (stateRewardsGetter(mecStatesToGlobalStates[i]) != optimizationDirectionFactor * mecStateRewardsGetter(i)) {
                cout << "Mismatch" << endl;
                abort();
            }
        }
    }*/ // Debug: check if state rewards are correct.

    /*{
        cout << "MEC row, global row, global rew, mec rew, rewards should match" << endl;
        for (uint64_t i = 0; i < mecRowsToGlobalRows.size(); i++) {
            cout << i << " " << mecRowsToGlobalRows[i] << " " << actionRewardsGetter(mecRowsToGlobalRows[i]) << " " << mecActionRewardsGetter(i) << endl;
            if (actionRewardsGetter(mecRowsToGlobalRows[i]) != optimizationDirectionFactor * mecActionRewardsGetter(i)) {
                cout << "Mismatch" << endl;
                abort();
            }
        }
    }*/ // Debug: check if action rewards are correct.

    // Initialized to always point at the first row of the row group.
    auto scheduler = std::vector<uint64_t>(mecMatrix.getRowGroupCount(), 0);
    for (uint64_t i = 0; i < mecMatrix.getRowGroupCount (); i++) {
        scheduler[i] = mecMatrix.getRowGroupIndices()[i];
    }
    auto [detMatrix, detActionRewardsGetter] = applyScheduler(mecMatrix, scheduler, mecActionRewardsGetter);
    auto detStateRewardsGetter = mecStateRewardsGetter; // Rename for clarity.
    auto helper = std::make_unique<storm::modelchecker::helper::SparseDeterministicInfiniteHorizonHelper<ValueType>>(detMatrix);
    /*{
        cout << "Scheduler:" << endl;
        cout << "mecState, choice" << endl;
        for (uint64_t i = 0; i < scheduler.size(); i++) {
            cout << i << " " << scheduler[i] << " " << endl;
        }
        cout << "detMatrix:" << endl;
        detMatrix.printAsMatlabMatrix(cout);

        cout << "det row, MEC row, det rew, MEC rew rewards should match" << endl;
        for (uint64_t i = 0; i < scheduler.size(); i++) {
            cout << i << " " << scheduler[i] << " " << detActionRewardsGetter(i) << " " << mecActionRewardsGetter(scheduler[i]) << endl;
            if (detActionRewardsGetter(i) != optimizationDirectionFactor * mecActionRewardsGetter(scheduler[i])) {
                cout << "Mismatch" << endl;
                abort();
            }
        }
    }*/ // Debug: check if applying the scheduler was a success, and check if the action rewards are still right

    vector<ValueType> gains(mecMatrix.getRowGroupCount());
    vector<ValueType> oldGains(mecMatrix.getRowGroupCount()); // For debugging purposes, the gains should always increase monotonically (i.e. never decrease).
    //vector<ValueType> biases(mecMatrix.getRowGroupCount());
    bool schedulerChanged = false;
    // For debugging purposes (loop detection) we keep track of the schedulers thus far.
    auto oldSchedulers = storm::storage::FlatSet<vector<uint64_t>>();

    int n = 0;
    while (true) {
        cout << n << " -----------------------------" << oldSchedulers.size() << endl; // For debugging
        /*{
            for (uint64_t state = 0; state < gains.size(); state++) {
                if (oldGains[state] * optimizationDirectionFactor > gains[state] * optimizationDirectionFactor) {
                    cout << "Gain got worse, this should never happen" << endl;
                    cout << this->getOptimizationDirection();
                    cout << "state " << state << "old: " << oldGains[state] << "new" << gains[state] << endl;
                }
            }
        }*/ // Debug: Check that the gain is indeed monotonically increasing.
        {
            for (auto const& oldSched : oldSchedulers) {
                bool sameSoFar = true;
                for (uint64_t state = 0; state < scheduler.size(); state++) {
                    //cout << "check scheduler repeated: " << scheduler[state] << oldSched[state] << endl;
                    if (oldSched[state] != scheduler[state]) {
                        sameSoFar = false;
                    }
                }
                if (sameSoFar) {
                    cout << "Scheduler repeating. This might be due to numerical instability, try --exact." << endl;
                    abort();
                }
            }
            oldSchedulers.insert(scheduler);
        } // Debug: keep track of old schedulers and check for scheduler repeating.
        std::tie(gains, biases) = helper->computeLraGainBias(env, detStateRewardsGetter, detActionRewardsGetter); // Line 2
        std::tie(schedulerChanged, scheduler) = gainImprovementStep(mecMatrix, scheduler, gains); // Line 3-4
        if (! schedulerChanged) {
            std::tie(schedulerChanged, scheduler) = biasImprovementStep(mecMatrix, scheduler, biases, mecStateRewardsGetter, mecActionRewardsGetter); // Line 3-4
        }
        if (schedulerChanged) {
            std::tie(detMatrix, detActionRewardsGetter) = applyScheduler(mecMatrix, scheduler, mecActionRewardsGetter);
            helper = std::make_unique<SparseDeterministicInfiniteHorizonHelper<ValueType>>(detMatrix);
            /*{
                cout << "Scheduler:" << endl;
                cout << "mecState, choice" << endl;
                for (uint64_t i = 0; i < scheduler.size(); i++) {
                    cout << i << " " << scheduler[i] << " " << endl;
                }
                cout << "detMatrix:" << endl;
                detMatrix.printAsMatlabMatrix(cout);

                cout << "det row, MEC row, det rew, MEC rew rewards should match" << endl;
                for (uint64_t i = 0; i < scheduler.size(); i++) {
                    cout << i << " " << scheduler[i] << " " << detActionRewardsGetter(i) << " " << mecActionRewardsGetter(scheduler[i]) << endl;
                    if (detActionRewardsGetter(i) != optimizationDirectionFactor * mecActionRewardsGetter(scheduler[i])) {
                        cout << "Mismatch" << endl;
                        abort();
                    }
                }
            }*/ // Debug: check if applying the scheduler was a success, and check if the action rewards are still right
            /*{
                cout << "Det state, global state, rewards should match" << endl;
                for (uint64_t i = 0; i < mecStatesToGlobalStates.size(); i++) {
                    cout << i << " " << mecStatesToGlobalStates[i] << " " << stateRewardsGetter(mecStatesToGlobalStates[i]) << " " << detStateRewardsGetter(i) << endl;
                    if (stateRewardsGetter(mecStatesToGlobalStates[i]) != optimizationDirectionFactor * detStateRewardsGetter(i)) {
                        cout << "Mismatch" << endl;
                        abort();
                    }
                }
            }*/ // Debug: check if state rewards are correct.
            n++;
        } else {
            ValueType exactResult = optimizationDirectionFactor * gains[0]; // We just return the first gain since they should all be the same in a MEC.
            {
                double valueIterationResult = storm::utility::convertNumber<double>(computeLraForMecVi(env, stateRewardsGetter, actionRewardsGetter, mec));
                double roundedResult = storm::utility::convertNumber<double>(exactResult);
                auto diff = roundedResult - valueIterationResult;
                if (diff > 1e-5 || diff < -1e-5) {
                    cout << "Wrong result compared to VI!" << endl;
                    cout << "VI: " << valueIterationResult << " PI: " << roundedResult << " DIR: " << this->getOptimizationDirection() << endl;
                    abort();
                } else {
                    cout << "Result OK" << endl;
                }
            } // Debug: compare final result to VI
            for (uint64_t i = 0; i < biases.size(); i++) { biases[i] = optimizationDirectionFactor * biases[i]; } // We might flip the sign on the biases in case we minimized.
            return exactResult;
        }
    }
}

/*!
 * Auxiliary function that adds the entries of the Ssp Matrix for a single choice (i.e., row)
 * Transitions that lead to a Component state will be redirected to a new auxiliary state (there is one aux. state for each component).
 * Transitions that don't lead to a Component state are copied (taking a state index mapping into account).
 */
template<typename ValueType>
void addSspMatrixChoice(uint64_t const& inputMatrixChoice, storm::storage::SparseMatrix<ValueType> const& inputTransitionMatrix,
                        std::vector<uint64_t> const& inputToSspStateMap, uint64_t const& numberOfNonComponentStates, uint64_t const& currentSspChoice,
                        storm::storage::SparseMatrixBuilder<ValueType>& sspMatrixBuilder) {
    // As there could be multiple transitions to the same MEC, we accumulate them in this map before adding them to the matrix builder.
    std::map<uint64_t, ValueType> auxiliaryStateToProbabilityMap;

    for (auto const& transition : inputTransitionMatrix.getRow(inputMatrixChoice)) {
        if (!storm::utility::isZero(transition.getValue())) {
            auto const& sspTransitionTarget = inputToSspStateMap[transition.getColumn()];
            // Since the auxiliary Component states are appended at the end of the matrix, we can use this check to
            // decide whether the transition leads to a component state or not
            if (sspTransitionTarget < numberOfNonComponentStates) {
                // If the target state is not contained in a component, we can copy over the entry.
                sspMatrixBuilder.addNextValue(currentSspChoice, sspTransitionTarget, transition.getValue());
            } else {
                // If the target state is contained in component i, we need to add the probability to the corresponding field in the vector
                // so that we are able to write the cumulative probability to the component into the matrix later.
                auto insertionRes = auxiliaryStateToProbabilityMap.emplace(sspTransitionTarget, transition.getValue());
                if (!insertionRes.second) {
                    // sspTransitionTarget already existed in the map, i.e., there already was a transition to that component.
                    // Hence, we add up the probabilities.
                    insertionRes.first->second += transition.getValue();
                }
            }
        }
    }

    // Now insert all (cumulative) probability values that target a component.
    for (auto const& componentToProbEntry : auxiliaryStateToProbabilityMap) {
        sspMatrixBuilder.addNextValue(currentSspChoice, componentToProbEntry.first, componentToProbEntry.second);
    }
}

template<typename ValueType>
std::pair<storm::storage::SparseMatrix<ValueType>, std::vector<ValueType>> SparseNondeterministicInfiniteHorizonHelper<ValueType>::buildSspMatrixVector(
    std::vector<ValueType> const& mecLraValues, std::vector<uint64_t> const& inputToSspStateMap, storm::storage::BitVector const& statesNotInComponent,
    uint64_t numberOfNonComponentStates, std::vector<std::pair<uint64_t, uint64_t>>* sspComponentExitChoicesToOriginalMap) {
    auto const& choiceIndices = this->_transitionMatrix.getRowGroupIndices();

    std::vector<ValueType> rhs;
    uint64_t numberOfSspStates = numberOfNonComponentStates + this->_longRunComponentDecomposition->size();
    storm::storage::SparseMatrixBuilder<ValueType> sspMatrixBuilder(0, numberOfSspStates, 0, true, true, numberOfSspStates);
    // If the source state of a transition is not contained in any component, we copy its choices (and perform the necessary modifications).
    uint64_t currentSspChoice = 0;
    for (auto nonComponentState : statesNotInComponent) {
        sspMatrixBuilder.newRowGroup(currentSspChoice);
        for (uint64_t choice = choiceIndices[nonComponentState]; choice < choiceIndices[nonComponentState + 1]; ++choice, ++currentSspChoice) {
            rhs.push_back(storm::utility::zero<ValueType>());
            addSspMatrixChoice(choice, this->_transitionMatrix, inputToSspStateMap, numberOfNonComponentStates, currentSspChoice, sspMatrixBuilder);
        }
    }
    // Now we construct the choices for the auxiliary states which reflect former Component states.
    for (uint64_t componentIndex = 0; componentIndex < this->_longRunComponentDecomposition->size(); ++componentIndex) {
        auto const& component = (*this->_longRunComponentDecomposition)[componentIndex];
        sspMatrixBuilder.newRowGroup(currentSspChoice);
        // For nondeterministic models it might still be that we leave the component again. This needs to be reflected in the SSP
        // by adding the "exiting" choices of the MEC to the axiliary states
        for (auto const& element : component) {
            uint64_t componentState = internal::getComponentElementState(element);
            for (uint64_t choice = choiceIndices[componentState]; choice < choiceIndices[componentState + 1]; ++choice) {
                // If the choice is not contained in the component itself, we have to add a similar distribution to the auxiliary state.
                if (!internal::componentElementChoicesContains(element, choice)) {
                    rhs.push_back(storm::utility::zero<ValueType>());
                    addSspMatrixChoice(choice, this->_transitionMatrix, inputToSspStateMap, numberOfNonComponentStates, currentSspChoice, sspMatrixBuilder);
                    if (sspComponentExitChoicesToOriginalMap) {
                        // Later we need to be able to map this choice back to the original input model
                        sspComponentExitChoicesToOriginalMap->emplace_back(componentState, choice - choiceIndices[componentState]);
                    }
                    ++currentSspChoice;
                }
            }
        }
        // For each auxiliary state, there is the option to achieve the reward value of the LRA associated with the component.
        rhs.push_back(mecLraValues[componentIndex]);
        if (sspComponentExitChoicesToOriginalMap) {
            // Insert some invalid values so we can later detect that this choice is not an exit choice
            sspComponentExitChoicesToOriginalMap->emplace_back(std::numeric_limits<uint_fast64_t>::max(), std::numeric_limits<uint_fast64_t>::max());
        }
        ++currentSspChoice;
    }
    return std::make_pair(sspMatrixBuilder.build(currentSspChoice, numberOfSspStates, numberOfSspStates), std::move(rhs));
}

template<typename ValueType>
void SparseNondeterministicInfiniteHorizonHelper<ValueType>::constructOptimalChoices(
    std::vector<uint64_t> const& sspChoices, storm::storage::SparseMatrix<ValueType> const& sspMatrix, std::vector<uint64_t> const& inputToSspStateMap,
    storm::storage::BitVector const& statesNotInComponent, uint64_t numberOfNonComponentStates,
    std::vector<std::pair<uint64_t, uint64_t>> const& sspComponentExitChoicesToOriginalMap) {
    // We first take care of non-mec states
    storm::utility::vector::setVectorValues(this->_producedOptimalChoices.get(), statesNotInComponent, sspChoices);
    // Secondly, we consider MEC states. There are 3 cases for each MEC state:
    // 1. The SSP choices encode that we want to stay in the MEC
    // 2. The SSP choices encode that we want to leave the MEC and
    //      a) we take an exit (non-MEC) choice at the given state
    //      b) we have to take a MEC choice at the given state in a way that eventually an exit state of the MEC is reached
    uint64_t exitChoiceOffset = sspMatrix.getRowGroupIndices()[numberOfNonComponentStates];
    for (auto const& mec : *this->_longRunComponentDecomposition) {
        // Get the sspState of this MEC (using one representative mec state)
        auto const& sspState = inputToSspStateMap[mec.begin()->first];
        uint64_t sspChoiceIndex = sspMatrix.getRowGroupIndices()[sspState] + sspChoices[sspState];
        // Obtain the state and choice of the original model to which the selected choice corresponds.
        auto const& originalStateChoice = sspComponentExitChoicesToOriginalMap[sspChoiceIndex - exitChoiceOffset];
        // Check if we are in Case 1 or 2
        if (originalStateChoice.first == std::numeric_limits<uint_fast64_t>::max()) {
            // The optimal choice is to stay in this mec (Case 1)
            // In this case, no further operations are necessary. The scheduler has already been set to the optimal choices during the call of computeLraForMec.
            STORM_LOG_ASSERT(sspMatrix.getRow(sspState, sspChoices[sspState]).getNumberOfEntries() == 0, "Expected empty row at choice that stays in MEC.");
        } else {
            // The best choice is to leave this MEC via the selected state and choice. (Case 2)
            // Set the exit choice (Case 2.a)
            this->_producedOptimalChoices.get()[originalStateChoice.first] = originalStateChoice.second;
            // The remaining states in this MEC need to reach the state with the exit choice with probability 1. (Case 2.b)
            // Perform a backwards search from the exit state, only using MEC choices
            // We start by setting an invalid choice to all remaining mec states (so that we can easily detect them as unprocessed)
            for (auto const& stateActions : mec) {
                if (stateActions.first != originalStateChoice.first) {
                    this->_producedOptimalChoices.get()[stateActions.first] = std::numeric_limits<uint64_t>::max();
                }
            }
            // Ensure that backwards transitions are available
            this->createBackwardTransitions();
            // Now start a backwards DFS
            std::vector<uint64_t> stack = {originalStateChoice.first};
            while (!stack.empty()) {
                uint64_t currentState = stack.back();
                stack.pop_back();
                for (auto const& backwardsTransition : this->_backwardTransitions->getRowGroup(currentState)) {
                    uint64_t predecessorState = backwardsTransition.getColumn();
                    if (mec.containsState(predecessorState)) {
                        auto& selectedPredChoice = this->_producedOptimalChoices.get()[predecessorState];
                        if (selectedPredChoice == std::numeric_limits<uint64_t>::max()) {
                            // We don't already have a choice for this predecessor.
                            // We now need to check whether there is a *MEC* choice leading to currentState
                            for (auto const& predChoice : mec.getChoicesForState(predecessorState)) {
                                for (auto const& forwardTransition : this->_transitionMatrix.getRow(predChoice)) {
                                    if (forwardTransition.getColumn() == currentState && !storm::utility::isZero(forwardTransition.getValue())) {
                                        // Playing this choice (infinitely often) will lead to current state (infinitely often)!
                                        selectedPredChoice = predChoice - this->_transitionMatrix.getRowGroupIndices()[predecessorState];
                                        stack.push_back(predecessorState);
                                        break;
                                    }
                                }
                                if (selectedPredChoice != std::numeric_limits<uint64_t>::max()) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename ValueType>
std::vector<ValueType> SparseNondeterministicInfiniteHorizonHelper<ValueType>::buildAndSolveSsp(Environment const& env,
                                                                                                std::vector<ValueType> const& componentLraValues) {
    STORM_LOG_ASSERT(this->_longRunComponentDecomposition != nullptr, "Decomposition not computed, yet.");

    // For fast transition rewriting, we build a mapping from the input state indices to the state indices of a new transition matrix
    // which redirects all transitions leading to a former component state to a new auxiliary state.
    // There will be one auxiliary state for each component. These states will be appended to the end of the matrix.

    // First gather the states that are part of a component
    // and create a mapping from states that lie in a component to the corresponding component index.
    storm::storage::BitVector statesInComponents(this->_transitionMatrix.getRowGroupCount());
    std::vector<uint64_t> inputToSspStateMap(this->_transitionMatrix.getRowGroupCount(), std::numeric_limits<uint64_t>::max());
    for (uint64_t currentComponentIndex = 0; currentComponentIndex < this->_longRunComponentDecomposition->size(); ++currentComponentIndex) {
        for (auto const& element : (*this->_longRunComponentDecomposition)[currentComponentIndex]) {
            uint64_t state = internal::getComponentElementState(element);
            statesInComponents.set(state);
            inputToSspStateMap[state] = currentComponentIndex;
        }
    }
    // Now take care of the non-component states. Note that the order of these states will be preserved.
    uint64_t numberOfNonComponentStates = 0;
    storm::storage::BitVector statesNotInComponent = ~statesInComponents;
    for (auto nonComponentState : statesNotInComponent) {
        inputToSspStateMap[nonComponentState] = numberOfNonComponentStates;
        ++numberOfNonComponentStates;
    }
    // Finalize the mapping for the component states which now still assigns component states to to their component index.
    // To make sure that they point to the auxiliary states (located at the end of the SspMatrix), we need to shift them by the
    // number of states that are not in a component.
    for (auto mecState : statesInComponents) {
        inputToSspStateMap[mecState] += numberOfNonComponentStates;
    }

    // For scheduler extraction, we will need to create a mapping between choices at the auxiliary states and the
    // corresponding choices in the original model.
    std::vector<std::pair<uint_fast64_t, uint_fast64_t>> sspComponentExitChoicesToOriginalMap;

    // The next step is to create the SSP matrix and the right-hand side of the SSP.
    auto sspMatrixVector = buildSspMatrixVector(componentLraValues, inputToSspStateMap, statesNotInComponent, numberOfNonComponentStates,
                                                this->isProduceSchedulerSet() ? &sspComponentExitChoicesToOriginalMap : nullptr);

    // Set-up a solver
    storm::solver::GeneralMinMaxLinearEquationSolverFactory<ValueType> minMaxLinearEquationSolverFactory;
    storm::solver::MinMaxLinearEquationSolverRequirements requirements =
        minMaxLinearEquationSolverFactory.getRequirements(env, true, true, this->getOptimizationDirection(), false, this->isProduceSchedulerSet());
    requirements.clearBounds();
    STORM_LOG_THROW(!requirements.hasEnabledCriticalRequirement(), storm::exceptions::UnmetRequirementException,
                    "Solver requirements " + requirements.getEnabledRequirementsAsString() + " not checked.");
    std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> solver = minMaxLinearEquationSolverFactory.create(env, sspMatrixVector.first);
    solver->setHasUniqueSolution();
    solver->setHasNoEndComponents();
    solver->setTrackScheduler(this->isProduceSchedulerSet());
    auto lowerUpperBounds = std::minmax_element(componentLraValues.begin(), componentLraValues.end());
    solver->setLowerBound(*lowerUpperBounds.first);
    solver->setUpperBound(*lowerUpperBounds.second);
    solver->setRequirementsChecked();

    // Solve the equation system
    std::vector<ValueType> x(sspMatrixVector.first.getRowGroupCount());
    solver->solveEquations(env, this->getOptimizationDirection(), x, sspMatrixVector.second);

    // Prepare scheduler (if requested)
    if (this->isProduceSchedulerSet() && solver->hasScheduler()) {
        // Translate result for ssp matrix to original model
        constructOptimalChoices(solver->getSchedulerChoices(), sspMatrixVector.first, inputToSspStateMap, statesNotInComponent, numberOfNonComponentStates,
                                sspComponentExitChoicesToOriginalMap);
    } else {
        STORM_LOG_ERROR_COND(!this->isProduceSchedulerSet(), "Requested to produce a scheduler, but no scheduler was generated.");
    }

    // Prepare result vector.
    // For efficiency reasons, we re-use the memory of our rhs for this!
    std::vector<ValueType> result = std::move(sspMatrixVector.second);
    result.resize(this->_transitionMatrix.getRowGroupCount());
    result.shrink_to_fit();
    storm::utility::vector::selectVectorValues(result, inputToSspStateMap, x);
    return result;
}

template class SparseNondeterministicInfiniteHorizonHelper<double>;
template class SparseNondeterministicInfiniteHorizonHelper<storm::RationalNumber>;

}  // namespace helper
}  // namespace modelchecker
}  // namespace storm